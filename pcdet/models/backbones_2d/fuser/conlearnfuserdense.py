import torch
import numpy as np
from torch import nn
from torch.nn.init import kaiming_normal_
from ...model_utils import centernet_utils
from ...model_utils import model_nms_utils
from ....utils import loss_utils
from ....utils.spconv_utils import replace_feature, spconv
import copy
from easydict import EasyDict
from spconv.core import ConvAlgo
import torchvision
import torchvision.ops as ops
from ... import dense_heads
from functools import partial
def build_dense_head(model_cfg):
    dense_head_module = dense_heads.__all__[model_cfg.NAME](
        model_cfg = model_cfg,
        input_channels = model_cfg.IN_CHANNEL,
        num_class = 10 if not model_cfg.CLASS_AGNOSTIC else 1, # for nus
        class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',\
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
        grid_size = None,
        point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        predict_boxes_when_training = True,
        voxel_size = [0.075, 0.075, 0.2]
    )

    return dense_head_module

class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False, norm_func=None):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels) if norm_func is None else norm_func(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CenterHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        norm_func = partial(nn.BatchNorm2d, eps=self.model_cfg.get('BN_EPS', 1e-5), momentum=self.model_cfg.get('BN_MOM', 0.1))
        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            norm_func(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                    norm_func=norm_func
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()
        ret_boxes_src = gt_boxes.new_zeros(num_max_objs, gt_boxes.shape[-1])
        ret_boxes_src[:gt_boxes.shape[0]] = gt_boxes

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask, ret_boxes_src

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': [],
            'target_boxes_src': [],
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, target_boxes_src_list = [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask, ret_boxes_src = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                target_boxes_src_list.append(ret_boxes_src.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['target_boxes_src'].append(torch.stack(target_boxes_src_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

            if 'iou' in pred_dict or self.model_cfg.get('IOU_REG_LOSS', False):

                batch_box_preds = centernet_utils.decode_bbox_from_pred_dicts(
                    pred_dict=pred_dict,
                    point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                    feature_map_stride=self.feature_map_stride
                )  # (B, H, W, 7 or 9)

                if 'iou' in pred_dict:
                    batch_box_preds_for_iou = batch_box_preds.permute(0, 3, 1, 2)  # (B, 7 or 9, H, W)

                    iou_loss = loss_utils.calculate_iou_loss_centerhead(
                        iou_preds=pred_dict['iou'],
                        batch_box_preds=batch_box_preds_for_iou.clone().detach(),
                        mask=target_dicts['masks'][idx],
                        ind=target_dicts['inds'][idx], gt_boxes=target_dicts['target_boxes_src'][idx]
                    )
                    loss += iou_loss
                    tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()

                if self.model_cfg.get('IOU_REG_LOSS', False):
                    iou_reg_loss = loss_utils.calculate_iou_reg_loss_centerhead(
                        batch_box_preds=batch_box_preds_for_iou,
                        mask=target_dicts['masks'][idx],
                        ind=target_dicts['inds'][idx], gt_boxes=target_dicts['target_boxes_src'][idx]
                    )
                    if target_dicts['masks'][idx].sum().item() != 0:
                        iou_reg_loss = iou_reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                        loss += iou_reg_loss
                        tb_dict['iou_reg_loss_head_%d' % idx] = iou_reg_loss.item()
                    else:
                        loss += (batch_box_preds_for_iou * 0.).sum()
                        tb_dict['iou_reg_loss_head_%d' % idx] = (batch_box_preds_for_iou * 0.).sum()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            batch_iou = (pred_dict['iou'] + 1) * 0.5 if 'iou' in pred_dict else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]

                if post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False) and 'pred_iou' in final_dict:
                    pred_iou = torch.clamp(final_dict['pred_iou'], min=0, max=1.0)
                    IOU_RECTIFIER = final_dict['pred_scores'].new_tensor(post_process_cfg.IOU_RECTIFIER)
                    final_dict['pred_scores'] = torch.pow(final_dict['pred_scores'], 1 - IOU_RECTIFIER[final_dict['pred_labels']]) * torch.pow(pred_iou, IOU_RECTIFIER[final_dict['pred_labels']])

                if post_process_cfg.NMS_CONFIG.NMS_TYPE not in  ['circle_nms', 'class_specific_nms']:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':
                    selected, selected_scores = model_nms_utils.class_specific_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None)
                    )
                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms':
                    raise NotImplementedError

                final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                final_dict['pred_scores'] = selected_scores
                final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts
        import pdb;pdb.set_trace()
        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict


class LidarCenterHeadRoI(CenterHead):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training = True):
        super().__init__(model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training)
        self.pc_start = model_cfg.PC_START
        self.out_stride = model_cfg.OUT_STRIDE
    
    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))
        
        targets_dict_con = {}
        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict
            targets_dict_con = self.centerpoint_roi_pool(data_dict, spatial_features_2d, training = True)

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            #targets_dict = {}
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            #if self.predict_boxes_when_training:
                #targets_dict_other = self.centerpoint_roi_pool(pred_dicts, spatial_features_2d)
                #targets_dict_train = self.centerpoint_roi_pool(data_dict, spatial_features_2d, training = True)
                # rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                # targets_dict['rois'] = rois
                # targets_dict['roi_scores'] = roi_scores
                # targets_dict['roi_labels'] = roi_labels
                # targets_dict['has_class_labels'] = True
            #else:
                #targets_dict_other['final_box_dicts'] = pred_dicts

        return data_dict, targets_dict_con
    
    def get_box_center(self, boxes, num_point = 5, training = False):
        '''
        boxs: [xs=1, ys=1, center_z=1, dim=3, angle=1, vel=2] ,[N,9]
        '''
        centers = []
        B_boxes = []
        if training:
            for i in range(boxes['gt_boxes'].shape[0]):
                B_boxes.append({'pred_boxes': boxes['gt_boxes'][i,:]})
        else:
            B_boxes = boxes
        for box in B_boxes: # box(dict): ['pred_boxes', 'pred_scores', 'pred_labels']
            box = box['pred_boxes']
            if num_point == 1 or len(box) == 0:
                centers.append(box[:, :3])

            elif num_point == 5:
                center2d = box[:, :2] 
                height = box[:, 2:3] 
                dim2d = box[:, 3:5] # 3-6? 可以进3维 这里存疑 zgx: (w, l, h)
                rotation_y = box[:, -3]

                corners = center_to_corner_box2d(center2d, dim2d, rotation_y)
                
                front_middle = torch.cat([(corners[:, 0] + corners[:, 1]) / 2, height], dim=-1)
                back_middle = torch.cat([(corners[:, 2] + corners[:, 3]) / 2, height], dim=-1)
                left_middle = torch.cat([(corners[:, 0] + corners[:, 3]) / 2, height], dim=-1)
                right_middle = torch.cat([(corners[:, 1] + corners[:, 2]) / 2, height], dim=-1) 
                
                points = torch.cat([box[:, :3], front_middle, back_middle, left_middle, \
                    right_middle], dim=0) # 5个点
                centers.append(points)
            else:
                raise NotImplementedError()
        
        return centers

    def get_interpolate_features(self, x, batch_centers, num_point=5):
        batch_size = x.shape[0]
        ret_maps = []

        for batch_idx in range(batch_size):
            xs, ys = self._absl_to_relative(batch_centers[batch_idx])
            
            # N x C
            feature_map = bilinear_interpolate_torch(x[batch_idx].permute(1, 2, 0), xs, ys)

            if num_point > 1:
                section_size = len(feature_map) // num_point
                feature_map = torch.cat([feature_map[i*section_size: (i+1)*section_size] for i in range(num_point)], dim=1)

            ret_maps.append(feature_map)

        return ret_maps
    
    def _absl_to_relative(self, absolute):
        a1 = (absolute[..., 0] - self.pc_start[0]) / self.voxel_size[0] / self.out_stride 
        a2 = (absolute[..., 1] - self.pc_start[1]) / self.voxel_size[1] / self.out_stride 

        return a1, a2


    def reorder_first_stage_pred_and_feature(self, first_pred, features, example = None):
        batch_size = len(first_pred)
        box_length = first_pred[0]['pred_boxes'].shape[1]

        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in first_pred])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error

        feature_vector_length = features[0].shape[-1] # roi features 的特征的维度

        # NMS_POST_MAXSIZE = first_pred[i]['pred_boxes'].shape[0] # MAX_POST_NMS

        rois = first_pred[0]['pred_boxes'].new_zeros((batch_size, 
            num_max_rois, box_length
        ))
        roi_scores = first_pred[0]['pred_scores'].new_zeros((batch_size,
            num_max_rois
        ))
        roi_labels = first_pred[0]['pred_labels'].new_zeros((batch_size,
            num_max_rois), dtype=torch.long
        )
        roi_features = features[0].new_zeros((batch_size, 
            num_max_rois, feature_vector_length 
        ))
        for i in range(batch_size):
            num_obj = features[i].shape[0]
            # basically move rotation to position 6, so now the box is 7 + C . C is 2 for nuscenes to
            # include velocity target

            box_preds = first_pred[i]['pred_boxes']

            # if self.roi_head.code_size == 9: 
            # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y
            box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 8, 6, 7]] #TODO 注意后续根据需求修改code_size

            rois[i, :num_obj] = box_preds
            roi_labels[i, :num_obj] = first_pred[i]['pred_labels'] + 1 #TODO 是否需要+1 根据需求来
            roi_scores[i, :num_obj] = first_pred[i]['pred_scores']
            roi_features[i, :num_obj] = features[i]

        example = {}
        example['rois'] = rois
        example['roi_labels'] = roi_labels
        example['roi_scores'] = roi_scores
        example['roi_features'] = roi_features # [bs, k, d_m]
        example['has_class_labels']= True
        return example



    def centerpoint_roi_pool(self, ret_dict, bev_feature, training = False):
        
        if self.training:
            batch_centers = self.get_box_center(boxes = ret_dict, training = True) # type(ret_dict) should not be list 
        else:
            batch_centers = self.get_box_center(boxes = ret_dict, training = False)
        # else:
        #     batch_centers = self.get_box_center(batch_dict['gt_boxes'])
        f = self.get_interpolate_features(x = bev_feature, batch_centers = batch_centers) # 和camera共用了head，这里的lidar_bev还没改名 
        if self.training:
            return f
        else:
            example = self.reorder_first_stage_pred_and_feature(first_pred = ret_dict, features = f)
            return example
    



def torch_to_np_dtype(ttype):
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2 ** ndim, ndim)
    return corners

def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    rot_mat_T = torch.stack([torch.stack([rot_cos, -rot_sin]), torch.stack([rot_sin, rot_cos])])
    return torch.einsum("aij,jka->aik", (points, rot_mat_T))
def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners

def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)
    Returns:
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans

class ConleanFuserDense(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.model_cfg.IMG_HEAD.IN_CHANNEL = 128
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
            )
        # self.img_box_net = ImgRoIHead() # 所有参数都有默认, 最后的大小取决于output_size, 默认7*7 # feiyang version
        # self.img_box_net = SeclectBox(model_cfg=self.model_cfg) # guoxin version
        if self.training:
            self.img_box_net = build_dense_head(self.model_cfg.IMG_HEAD) # guoxin  v2 # build center_head_roi
            # self.lidar_box_net = SeclectBox(model_cfg = self.model_cfg)
            self.lidar_box_net = LidarCenterHeadRoI(model_cfg = self.model_cfg.LIDAR_HEAD, 
                                                      input_channels = 128, 
                                                      num_class = 10, 
                                                      class_names =  ['car','truck', 'construction_vehicle', 'bus', 'trailer','barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'], 
                                                      grid_size = None, 
                                                      point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
                                                      predict_boxes_when_training = True, 
                                                      voxel_size = [0.075, 0.075, 0.2])
        self.point_cloud_range = torch.Tensor([-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]).cuda()


    def shoot(self, lidar_targets_dict,camera_targets_dict,threshol = 0.1, kd_tree_neigbbor=8):
        '''
        lidar_targets_dict: ['rois'] ['roi_labels'] ['roi_scores'] ['roi_features'] ['has_class_labels']
            ['rois']: [bs, n, 9], 9分别是x, y, z, w, l, h, rotation_y, velocity_x, velocity_y
            ['roi_labels']: [bs, n], cls id
            ['roi_scores']: [bs, n], 0-1评分, 从大到小排序
            ['roi_features']: [bs, n, 5*c], c是128
            ['has_class_labels']: True
        camera_targets_dict: 相同, 第二维m接近500, 一般都大于n
        '''
        
        from ....ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu,boxes_iou_bev
        from scipy.spatial import cKDTree
        lidar_roi_box=lidar_targets_dict['rois']
        camera_roi_box=camera_targets_dict['rois']
        assert lidar_roi_box.shape[0]==camera_roi_box.shape[0]
        assert lidar_roi_box.shape[-1]==camera_roi_box.shape[-1]
        batch_size=lidar_roi_box.shape[0]
        device=lidar_roi_box.device
        ### 根据iou获得相近的box
        shoot_indices=[]
        for i in range(batch_size):
            ### boxes_iou3d_gpu 命中的极少 不对 
            # rets.append(boxes_iou3d_gpu(lidar_roi_box[i][:, 0:7], camera_roi_box[i][:, 0:7])) 
            ### boxes_iou_bev 命中的数量跟lidar的数量基本一致 但是计算出来的值不是0-1的 
            ### img=498 lidar=155 (rets1[0]>0.999999).sum()=tensor(155, device='cuda:0')
            ### img=498 lidar=157 (rets1[0]>0.999999).sum()=tensor(117, device='cuda:0')
            ret=boxes_iou_bev(lidar_roi_box[i][:, 0:7], camera_roi_box[i][:, 0:7])
            indices = torch.nonzero(ret > threshol)
            if indices.size(0) == 0:
                indices = torch.nonzero(ret > 0)
            else:
                pass
            # import pdb;pdb.set_trace()
            result = torch.zeros(indices.size(0), 3)
            result[:, 0] = indices[:, 0]
            result[:, 1] = indices[:, 1]
            result[:, 2] = ret[indices[:, 0], indices[:, 1]]
            result,_ = torch.sort(result, dim=0, descending=True) # 按照第三列从大到小排序结果是
            shoot_indices.append(result.to(device))
        ### 取出shoot到的box
        # 发现shoot_indices有为空的情况
        rois_shoot=[]
        roi_labels_shoot=[]
        roi_scores_shoot=[]
        roi_features_shoot=[]
        roi_features_neighbor=[]
        for i in range(batch_size):
            
            shoot_indice_l=shoot_indices[i][:, 0].int()
            shoot_indice_c=shoot_indices[i][:, 1].int()
            # print(i,shoot_indice_l,shoot_indice_c)
            rois_shoot.append(torch.index_select(lidar_targets_dict['rois'][i],0,shoot_indice_l).to(device))
            rois_shoot.append(torch.index_select(camera_targets_dict['rois'][i],0,shoot_indice_c).to(device))
            roi_labels_shoot.append(torch.index_select(lidar_targets_dict['roi_labels'][i],0,shoot_indice_l).to(device))
            roi_labels_shoot.append(torch.index_select(camera_targets_dict['roi_labels'][i],0,shoot_indice_c).to(device))
            roi_scores_shoot.append(torch.index_select(lidar_targets_dict['roi_scores'][i],0,shoot_indice_l).to(device))
            roi_scores_shoot.append(torch.index_select(camera_targets_dict['roi_scores'][i],0,shoot_indice_c).to(device))
            roi_features_shoot.append(torch.index_select(lidar_targets_dict['roi_features'][i],0,shoot_indice_l).to(device))
            roi_features_shoot.append(torch.index_select(camera_targets_dict['roi_features'][i],0,shoot_indice_c).to(device))
            kdtree_l = cKDTree(lidar_targets_dict['rois'][i][:,:2].cpu().detach().numpy())
            kdtree_c = cKDTree(camera_targets_dict['rois'][i][:,:2].cpu().detach().numpy())
            roi_features_neighbor_l=[]
            for j in rois_shoot[-2]: #选中lidar 从camera里查找neighbor作为负样本
                distances, neighbor_indices = kdtree_c.query(j[:2].cpu().detach().numpy(), k=kd_tree_neigbbor+1)
                roi_features_neighbor_l.append(torch.index_select(camera_targets_dict['roi_features'][i],0,torch.tensor(neighbor_indices).to(device)))
            if len(roi_features_neighbor_l) == 0:
                import pdb; pdb.set_trace()
            roi_features_neighbor.append(torch.stack(roi_features_neighbor_l))
            roi_features_neighbor_c=[]
            for j in rois_shoot[-1]:
                distances, neighbor_indices = kdtree_l.query(j[:2].cpu().detach().numpy(), k=kd_tree_neigbbor+1)
                roi_features_neighbor_c.append(torch.index_select(lidar_targets_dict['roi_features'][i],0,torch.tensor(neighbor_indices).to(device)))
            roi_features_neighbor.append(torch.stack(roi_features_neighbor_c))
        lidar_targets_dict_shoot={}
        lidar_targets_dict_shoot['rois']=rois_shoot[::2]
        lidar_targets_dict_shoot['roi_labels']=roi_labels_shoot[::2]
        lidar_targets_dict_shoot['roi_scores']=roi_scores_shoot[::2]
        lidar_targets_dict_shoot['roi_features']=roi_features_shoot[::2] #外层list len=bs # 内层tensor [n, 5*c]
        lidar_targets_dict_shoot['roi_features_kdtree']=roi_features_neighbor[::2] #外层list len=bs 每个tensor的shape[0]不一致 没法tensor # 内层tensor [n,9,5*c]，n个命中，第二维第1个点是命中的点本身，其他8是neighbor
        lidar_targets_dict_shoot['has_class_labels']= True
        camera_targets_dict_shoot={}
        camera_targets_dict_shoot['rois']=rois_shoot[1::2]
        camera_targets_dict_shoot['roi_labels']=roi_labels_shoot[1::2]
        camera_targets_dict_shoot['roi_scores']=roi_scores_shoot[1::2]
        camera_targets_dict_shoot['roi_features']=roi_features_shoot[1::2]
        camera_targets_dict_shoot['roi_features_kdtree']=roi_features_neighbor[1::2]
        camera_targets_dict_shoot['has_class_labels']= True
        
        return lidar_targets_dict_shoot,camera_targets_dict_shoot

    def train_shoot(self, batch_dict):
        pass

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality
                encoded_spconv_tensor: 
        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after multi-modal fusion
        """
        img_bev = batch_dict['spatial_features_img'] #[1, 80, 180, 180] (b, c, y, x)
        lidar_bev = batch_dict['spatial_features'] #[1, 128, 180, 180]
        #x = batch_dict['encoded_spconv_tensor'] # lidar branch 2d
        #import pdb;pdb.set_trace()
        #if self.training:
        batch_dict, lidar_targets_dict = self.lidar_box_net(batch_dict)
        batch_dict, camera_targets_dict = self.img_box_net(batch_dict)
        #import pdb;pdb.set_trace()
        #else:
           # batch_dict, lidar_targets_dict = self.lidar_box_net(batch_dict)
        #batch_dict, lidar_targets_dict = self.lidar_box_net(batch_dict) # x: sparse; lidar_bev: dense
        #import pdb;pdb.set_trace()
        # batch_dict['img_box'], batch_dict['img_box_indices'] = self.img_box_net(img_bev) # feiyang version
        #batch_dict, camera_targets_dict = self.img_box_net(batch_dict) # guoxin version
        # lidar_targets_dict camera_targets_dict
        

        # 对比学习
        if self.training:
            batch_dict['con_loss'] = loss_utils.InfoNCE_conloss(lidar_targets_dict,camera_targets_dict)
            #import pdb;pdb.set_trace()
            lidar_pred_loss, _ = self.lidar_box_net.get_loss()
        #import pdb;pdb.set_trace()
            img_pred_loss, _ = self.img_box_net.get_loss()
            batch_dict['con_loss'] = batch_dict['con_loss'] + lidar_pred_loss + img_pred_loss

        cat_bev = torch.cat([img_bev, lidar_bev],dim=1)
        mm_bev = self.conv(cat_bev)
        batch_dict['spatial_features'] = mm_bev
        return batch_dict