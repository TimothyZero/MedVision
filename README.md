## Medical Image Vision Operators

### 描述 (Description)

为了方便3D医学图像相关的任务, 实现了主要的计算机视觉算子,例如RoIAlign, NMS 等,并且基于这些算子实现了多种数据增强的方法.

Computer vision operators in medical image, such as RoIAlign, DCNv1, DCNv2 and NMS for both 2/3D images. And based on these operators, a variety of data augmentation methods have been implemented.

### 支持的CUDA算子 (Supported CUDA Operators)

> see medvision/csrc and medvision/ops

| Methods                | Torch | Cuda | Cpu | FP16 |
| ---------------------- | ----- | ---- | --- | ---- |
| RoI Align              | 2/3D  | 2/3D |     | yes  |
| RoI Align Rotated      |       | 2/3D |     | yes  |
| BBox overlaps          | 2/3D  | 2/3D |     | yes  |
| NMS                    | 2/3D  | 2/3D |     | yes  |
| soft-NMS               | 2/3D  |      | 2D  | yes  |
| DCN v1                 |       | 2/3D |     | yes  |
| DCN v2                 |       | 2/3D |     | yes  |
| Deformable RoI Pooling |       | 2D   |     | yes  |

**Torch** : *implemented with torch functions.*

### 支持的CUDA数据增强 (Supported CUDA Augmentations)

> see medvision/aug_cuda and medvision/aug_cuda_batch
>
> or medvision/aug_cpu for numpy version

| Methods                  | Cuda | FP16 | Ops               | CPU time/ms | GPU time/ms |
| ------------------------ | ---- | ---- | ----------------- | ----------- | ----------- |
| RandomAffine             | 2/3D | yes  | RoI Align Rotated | 5302        | 20          |
| RandomScale              | 2/3D | yes  | RoI Align Rotated | 5109        | 19          |
| RandomShift              | 2/3D | yes  | RoI Align Rotated | 2355        | 19          |
| RandomRotate             | 2/3D | yes  | RoI Align Rotated | 1164        | 20          |
| RandomFlip               | 2/3D | yes  |                   | 64          | 8           |
| CropRandom Series        | 2/3D | yes  | RoI Align Rotated | 56          | 1           |
| RandomElasticDeformation | 2/3D | yes  | DCN               | 12163       | 614         |
| Resize                   | 2/3D | yes  | RoI Align Rotated | 6365        | 12          |
| Pad                      | 2/3D | yes  |                   | 335         | 2           |
| Normalize +              | 2/3D | yes  |                   |             |             |
| RandomBlur               | 2/3D | yes  | Conv              | 450         | 3           |
| RandomNoise              | 2/3D | yes  |                   | 581         | 9           |
| Display                  | 2/3D | yes  |                   |             |             |
| Viewer                   | 2/3D | yes  |                   |             |             |

**All of these will support forward and backward (need test).**


### 可视化 (Visualization)

| Methods                  | Cuda                                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------------- |
| RandomAffine             | ![RandomAffine2D](./assets/RandomAffine2D.png) <br> ![RandomAffine3D](./assets/RandomAffine3D.png)            |
| RandomScale              | ![RandomScale3D](./assets/RandomScale3D.png)                                                                  |
| RandomShift              | ![RandomShift3D](./assets/RandomShift3D.png)                                                                  |
| RandomRotate             | ![](./assets/RandomRotate3D.gif)                                                                              |
| RandomFlip               | ![RandomFlip3D](./assets/RandomFlip3D.png)                                                                    |
| CropRandom Series        | ![CropRandomWithAffine3D](./assets/CropRandomWithAffine3D.gif)                                                |
| RandomElasticDeformation | ![raw](./assets/RandomElasticDeformationFast3D_raw.png) <br> ![](./assets/RandomElasticDeformationFast3D.png) |
| Resize                   | ![Resize3D](./assets/Resize3D.png)                                                                            |
| Pad                      | ![Pad3D](./assets/Pad3D.png)                                                                                  |
| RandomBlur               | ![RandomBlur2D](./assets/RandomBlur2D.png)  <br> ![RandomBlur3D](./assets/RandomBlur3D.png)                   |
| RandomNoise              | ![RandomNoise3D](./assets/RandomNoise3D.png)                                                                  |

### 待完成 (TODO)

- 


### 安装 (Installation)

```shell
# run and check you cuda and torch
# make sure your torch.version.cuda == cuda_home_version
python -m torch.utils.collect_env

# if needed
export PATH="path_to/gcc-5.4+/bin/:$PATH"
export CUDA_HOME="/path_to/cuda-9.0+"

git clone https://github.com/TimothyZero/MedVision
cd MedVision
python setup.py develop  # recommended
# or
pip install -e .  # -e : editable, sometimes may cause cpu 100%
```


### 测试环境 (Tested Environment)

| cuda | torch  | gcc  |                      |
| ---- | ------ | ---- | -------------------- |
| 9.0  | 1.6.0  | 5.4  | :white_check_mark:   |
| 10.1 | 1.7.1  | 7.5  | :white_check_mark:   |
| 10.2 | 1.8.1  | 7.5  | :white_check_mark:   |
| 11.1 | 1.11.0 | 7.5  | :white_check_mark:   |
| 11.8 | 2.0.0  | 11.4 | :white_check_mark: * |

`*: kaggle T4`

### 提示 (Tips)

1. `AT_CHECK` was not declared in this scope

For torch 1.5+, `AT_CHECK` is replaced with `TORCH_CHECK`, so if your torch version > 1.5 ,
```cuda
#ifndef AT_CHECK
#define AT_CHECK TORCH_CHECK
#endif
```
at the beginning of the .cu code.

2. debug in CUDA
```c
#include <stdio.h>

printf("Hello from block %d, thread %d\n", a, b);
```

3. `.contiguous()` is very import in roi align!

4. CUDA Too many resources requested for launch

   "Too Many Resources Requested for Launch - This error means that the number of registers available on the multiprocessor is being exceeded. Reduce the number of threads per block to solve the problem."

### 许可证 (License)

This framework is published under the Apache License Version 2.0.

### 致谢 (Acknowledge)

https://github.com/XinyiYing/D3Dnet

https://github.com/open-mmlab/mmdetection

https://github.com/MIC-DKFZ/medicaldetectiontoolkit

