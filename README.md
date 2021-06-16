### Description

Medical Image Vision Operators


### Supports 

##### cuda
- nms 2/3d
- roi align 2d/3d
- deformable convolution 2d/3d

##### cpu
- softnms 2d

### Installation

```shell
git clone ...
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

