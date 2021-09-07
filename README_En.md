## Medical Image Vision Operators

### Description

Computer vision operators in medical image, such as RoIAlign, DCNv1, DCNv2 and NMS for both 2/3D images.

### Supported CUDA Operators

| Methods                | Torch | Cuda | Cpu  | FP16 |
| ---------------------- | ----- | ---- | ---- | ---- |
| RoI Align              | 2/3D  | 2/3D |      | yes  |
| RoI Align Rotated      |       | 2/3D |      | yes  |
| BBox overlaps          | 2/3D  | 2/3D |      | yes  |
| NMS                    | 2/3D  | 2/3D |      | yes  |
| soft-NMS               | 2/3D  |      | 2D   | yes  |
| DCN v1                 |       | 2/3D |      | yes  |
| DCN v2                 |       | 2/3D |      | yes  |
| Deformable RoI Pooling |       | 2D   |      | yes  |


***Torch** : implemented with torch functions.*


### Supported CUDA Augmentations
| Methods                  | Cuda | FP16 | Ops               |
| ------------------------ | ---- | ---- | ----------------- |
| RandomAffine             | 2/3D | yes  | RoI Align Rotated |
| RandomScale              | 2/3D | yes  | RoI Align Rotated |
| RandomShift              | 2/3D | yes  | RoI Align Rotated |
| RandomRotate             | 2/3D | yes  | RoI Align Rotated |
| RandomFlip               | 2/3D | yes  |                   |
| CropRandom Series        | 2/3D | yes  | RoI Align Rotated |
| RandomElasticDeformation | 2/3D | yes  | DCN               |
| Resize                   | 2/3D | yes  | RoI Align Rotated |
| Pad                      | 2/3D | yes  |                   |
| Normalize +              | 2/3D | yes  |                   |
| RandomBlur               | 2/3D | yes  | Conv              |
| RandomNoise              | 2/3D | yes  |                   |
| Display                  | 2/3D | yes  |                   |
| Viewer                   | 2/3D | yes  |                   |

**All of these will support forward and backward (need test).**


### TODO

- [ ] saver


### Installation

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


### Tested Environment

```
gcc    5.4,   7.5
torch  1.6.0, 1.7.1, 1.8.1
cuda   9.0,   10.1,  10.2
```

### Tips

1. ‘AT_CHECK’ was not declared in this scope

For torch 1.5+, AT_CHECK is replaced with TORCH_CHECK, so if your torch version > 1.5 ,
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

### License

This framework is published under the Apache License Version 2.0.

### Acknowledge

https://github.com/XinyiYing/D3Dnet

https://github.com/open-mmlab/mmdetection

https://github.com/MIC-DKFZ/medicaldetectiontoolkit

