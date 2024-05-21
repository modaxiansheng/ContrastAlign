<div align="center">   

  # ContrastAlign: Toward Robust BEV Feature Alignment via Contrastive Learning for Multi-Modal 3D Object Detection
</div>

<div align="center">
  <img src="fig/vis.gif" />
</div>

<div align="justify">  

  This is the official repository of **ContrastAlign**. **ContrastAlign** utilizes contrastive learning to enhance the alignment of heterogeneous modalities, thereby improving the robustness of LiDAR-Camera BEV feature fusion.  we propose the L-Instance module, which directly outputs LiDAR instance features within LiDAR BEV features. Then, we introduce the C-Instance module, which predicts Camera instance features through RoI (Region of Interest) Pooling on Camera BEV features. The LiDAR instance features are then projected onto image instance features, and contrastive learning is employed to generate similar instance features between LiDAR and camera. Subsequently, through graph matching, neighboring camera instance features are matched to calculate similarity and construct positive and negative samples. During inference, the aligned features with high similarity in neighboring camera instance features are selected as alignment features to achieve BEV feature alignment.

</div>

------

<div align="justify">  

:fire: Contributions:
* We propose a robust fusion framework, named **ContrastAlign**, to address feature misalignment arising from projection errors between LiDAR and camera inputs.
* we propose the L-Instance and C-Instance modules, which directly outputs LiDAR and Camera instance features within LiDAR and Camera BEV features.  We propose InstanceFusion module, which utilizes contrastive learning to generate similarity instance features across heterogeneous modalities, and then use graph matching to calculate the similarity between the neighbor Camera instance features and the similarity instance features to complete the alignment of instance features.
* Extensive experiments validate the effectiveness of our ContrastAlign, demonstrating competitive performance on nuScenes. Notably, ContrastAlign maintains comparable performance across both clean settings and misaligned noisy conditions.

</div>

# Abstract

<div align="justify"> 

In the field of 3D object detection tasks, fusing heterogeneous modal (LiDAR and Camera) features into a unified Bird's Eye View (BEV) representation is a widely adopted paradigm. However, current methods are vulnerable to the impact of inaccurate calibration between LiDAR and camera sensors, resulting in **feature misalignment** issues during LiDAR-Camera BEV feature fusion. In this work, we propose **ContrastAlign**, which utilizes contrastive learning to enhance the alignment of heterogeneous modalities, thereby improving the robustness of LiDAR-Camera BEV feature fusion. Specifically, we propose the L-Instance module, which directly outputs LiDAR instance features within LiDAR BEV features. Then, we introduce the C-Instance module, which predicts Camera instance features through RoI (Region of Interest) Pooling on Camera BEV features. We utilize contrastive learning to generate similarity instance features across heterogeneous modalities, and then use graph matching to calculate the similarity between the neighbor Camera instance features and the similarity instance features to complete the alignment of instance features. Our method achieves SOTA (state-of-the-art) performance, with an mAP of 70.3%, surpassing BEVFusion by 1.8% on the nuScnes validation set. Importantly, our method outperforms BEVFusion by 8.4% under conditions with misalignment noise.

</div>

# Method

<div align="center">
  <img src="fig/main.png" />
</div>

<div align="justify">

Overview of our **ContrastAlign** framework. It is noteworthy that, to better preserve our core contributions, we simplified the overview. For detailed modules such as LSS, please refer to BEVFusion. We follow the baselines (bevfusion-mit, Transfusion} to generate LiDAR BEV features in the LiDAR branch. Then we propose the L-Instance module, which utilizes a 3D CNN backbone network to directly generate sparse LiDAR instance features (3D RoI features) similar to VoxelNeXt. The Camera BEV features are generated through the LSS in the Camera-to-BEV process. Subsequently, we propose the C-Instance module, which utilizes RoI pooling to produce Camera instance features (2D RoI features). We establish neighbor relationships among instance features by employing graph matching techniques. Subsequently, we leverage contrastive learning to create pairs of positive and negative samples, facilitating the learning of similarities across diverse modalities. In the inference phase, we retrieve aligned image instance features by querying based on their shared characteristics. Finally, we employ a dense detection head to accomplish the 3D detection task.

</div>

# Model Zoo

* Results on nuScenes **val set** under **noisy misalignment** setting.

| Method | Modality | NDS⬆️ | mAP⬆️ | Config |
| :---: | :---: | :---: | :---: | :---: |
| BEVfusion-MIT | LC | 65.7 | 60.8 | [config](tools/cfgs/nuscenes_models/bevfusion.yaml) |
| ContrastAlign | LC | 70.9 | 68.1 | [config](tools/cfgs/nuscenes_models/contrastalign.yaml) |

* Results on nuScenes **val set** under **clean** setting.

| Method | Modality | NDS⬆️| mAP⬆️ |
| :---: | :---: | :---: | :---: |
| BEVfusion-MIT | LC | 71.4 | 68.5 |
| ContrastAlign | LC | 72.5 | 70.3 |

* Results on nuScenes **test set**.

| Method | Modality | NDS⬆️| mAP⬆️ 
| :---: | :---: | :---: | :---: |
| BEVfusion-MIT | LC | 72.9 | 70.2 |
| ContrastAlign | LC | 73.8 | 71.8 |

# Dataset Preparation

**NuScenes Dataset** : Please download the [official NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and organize the downloaded files as follows:

```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```

Install the `nuscenes-devkit` with version `1.0.5` by running the following command:

```bash
pip install nuscenes-devkit==1.0.5
```

Generate the data infos (for multi-modal setting) by running the following command (it may take several hours):

```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval \
    --with_cam
```

# How to introduce misalignment noise into ContrastAlign

If you want to introduce misalignment noise into ContrastAlign, please modify the following settings:
```
MODEL:
    ...
    VTRANSFORM:
        NAME: DepthLSSTransform
        ...
        Noise: False

...
```
Noisy Misalignment will be introduced when **Noise** is True.

Whether Noisy Misalignment is introduced will be judged by the following [code](pcdet/models/view_transforms/depth_lss.py) :
```python
...
if not self.training:#test
    if self.noise:
        print("spatial_alignment_noise")
        lidar2image=self.spatial_alignment_noise(lidar2image,5)
        camera2lidar=self.spatial_alignment_noise(camera2lidar,5)
    else:
        print("clean")
...
```

Function **spatial_alignment_noise** is as following [code](pcdet/models/view_transforms/depth_lss.py) :
```python
def spatial_alignment_noise(self, ori_pose, severity):
    '''
    input: ori_pose 4*4
    output: noise_pose 4*4
    '''
    ct = [0.02, 0.04, 0.06, 0.08, 0.10][severity-1]*2
    cr = [0.002, 0.004, 0.006, 0.008, 0.10][severity-1]*2
    r_noise = torch.randn((3, 3), device=ori_pose.device)* cr
    t_noise = torch.randn((3), device=ori_pose.device) * ct
    ori_pose[..., :3, :3] += r_noise
    ori_pose[..., :3, 3]+= t_noise
    return ori_pose
```

# Requirements

All the codes are tested in the following environment:

* Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04)
* Python 3.8+
* torch                     1.12.1+cu113
* torchaudio              0.12.1+cu113
* torchvision              0.13.1+cu113
* scipy                     1.10.1
* spconv-cu113              2.3.6

All codes are developed based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) .

# Train and Inference

* Training is conducted on 8 NVIDIA GeForce RTX 3090 24G GPUs. 
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29535  train.py --launcher pytorch --batch_size 24  --extra_tag ContrastAlign_train --cfg_file cfgs/nuscenes_models/contrastalign.yaml  --save_to_file 
```

* During inference, we **remove** Test Time Augmentation (TTA) data augmentation, and the batch size is set to 1 on an A100 GPU.
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29541 test.py --launcher pytorch --batch_size 1 --extra_tag ContrastAlign_test --cfg_file cfgs/nuscenes_models/contrastalign.yaml --start_epoch 1 --eval_all --save_to_file --ckpt_dir ../output/nuscenes_models/contrastalign/contrastalign_train/ckpt
```

* All latency measurements are taken on the same workstation with an A100 GPU.

# Acknowledgement
Many thanks to these excellent open source projects:
- [BEVFusion-MIT](https://github.com/mit-han-lab/bevfusion) 
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [TransFusion](https://github.com/XuyangBai/TransFusion/) 
