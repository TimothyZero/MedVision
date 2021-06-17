## Medical Image Vision Operators

### Description

Computer vision operators in medical image, such as RoIAlign, DCN and NMS for both 2/3D images.

### Supported

Methods | Cuda | Cpu
---|---|---
RoI Align | 2/3D | 
NMS | 2/3D | 
DCNv1 | 2/3D |
DCNv2 | 2D |
Deformable RoI Pooling | 2D | 
soft-NMS | | 2D


### Installation

```shell
git clone https://github.com/TimothyZero/MedVision
pip install .
```


### Tested Environment

```
gcc 5.4, 7.5
torch 1.6.0, 1.7.1, 1.8.1
cuda 9.0, 10.1, 10.2
```

### Some issues

1. ‘AT_CHECK’ was not declared in this scope

For torch 1.5+, AT_CHECK is replaced with TORCH_CHECK, so if your torch is higher, 
```cuda
#define AT_CHECK TORCH_CHECK
```
at the beginning of the .cu code.

### License

This framework is published under the Apache License Version 2.0.

### Acknowledge

https://github.com/XinyiYing/D3Dnet

https://github.com/open-mmlab/mmdetection

https://github.com/MIC-DKFZ/medicaldetectiontoolkit

