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

# -*- coding:utf-8 -*-

import os
import pickle
import sys
import argparse

from vtktools import vtkShow
from vtkViewClass import vtkWindowView


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='tmp.pkl', type=str)
    parser.add_argument('--mode', default=1, choices=[1, 2, 3], type=int)
    args = parser.parse_args()

    try:
        if os.path.isabs(args.file):
            filepath = os.path.abspath(args.file)
        else:
            filepath = os.path.join(os.path.dirname(__file__), args.file)
        print(f'Visualizing {filepath} in mode {args.mode}')

        loaded_data = pickle.load(open(filepath, 'rb'))
        data, spacing = loaded_data['data'], loaded_data['spacing']

        if int(args.mode) == 1:
            vtkShow(data, spacing=spacing)
        elif int(args.mode) == 2:
            vtkWindowView(data, spacing=spacing)

        os.remove(os.path.dirname(__file__) + '/tmp.pkl')

    except Exception as e:
        print(e)
