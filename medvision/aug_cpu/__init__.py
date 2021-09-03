#  Copyright (c) 2020. The Medical Image Computing (MIC) Lab, 陶豪毅
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# TODO solve big image transform
from .loading import LoadPrepare, \
    LoadImageFromFile, LoadAnnotations, LoadCoordinate, \
    LoadWeights, \
    LoadPseudoAsSeg, LoadSegAsImg, \
    LoadPredictions
from .saving import SaveImageToFile, SaveAnnotations, SaveFolder, SplitPatches
from .aug_intensity import RGB2Gray, Gray2RGB, ChannelSelect,\
    AnnotationMap, \
    Normalize, MultiNormalize, AutoNormalize, \
    ImageErosion, ImageDilation, ImageClosing, ImageOpening, \
    RandomBlur, RandomGamma, RandomNoise, RandomSpike, RandomBiasField, \
    RandomCutout, ForegroundCutout, \
    CLAHE, \
    LoadCannyEdge, LoadSkeleton # LoadGradient
from .aug_spatial import Resize, Pad, \
    RandomScale, RandomRotate, RandomShift, RandomFlip, \
    RandomElasticDeformation, \
    RandomCrop, WeightedCrop, CenterCrop, \
    ForegroundCrop, FirstDetCrop, OnlyFirstDetCrop, \
    ForegroundPatches
from .aug_other import Repeat
from .aug_testtime import Patches, MultiScale, MultiGamma
from .custom import MultiGammaEns
from .viewer import Viewer, Display
from .formating import ToTensor, Collect
from .compose import ForwardCompose, OneOf, BackwardCompose
from .statistic import Property
