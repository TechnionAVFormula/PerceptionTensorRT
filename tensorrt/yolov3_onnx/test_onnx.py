#!/usr/bin/env python2
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

from __future__ import print_function

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import csv

import torch
import torchvision
import onnx
import caffe2.python.onnx.backend as backend

import sys, os
from test_utils import TestRunner

class OnnxTestRunner(TestRunner):
    def __init__(self, onnx_file_path, input_file_path, input_resolution, out_image, train_csv):
        super().__init__(input_file_path, input_resolution, out_image, train_csv)
        self.onnx_file_path = onnx_file_path
        self.model = None
        self.rep = None

    def _initialize(self):
        self.model = OnnxTestRunner.get_model(self.onnx_file_path)

        print("Preparing backend")
        self.rep = backend.prepare(self.model, device="CUDA:0") # or "CPU"

    def _infer(self, input_data):
        return self.rep.run(input_data)

    @staticmethod
    def get_model(model_path):
        print(f"Loading onnx {model_path}")
        model = onnx.load(model_path)
        print(f"Checking model")
        onnx.checker.check_model(model)
        return model

def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='enable verbose output (for debugging)')
    parser.add_argument('onnx', type=str)
    parser.add_argument('input_img', type=str)
    parser.add_argument('--train-csv', '-t', type=str)
    parser.add_argument('output_img', type=str)
    args = parser.parse_args()

    # Store the shape of the original input image in WH format, we will need it for later
    # Output shapes expected by the post-processor
    # Do inference with pyTorch on the onnx model

    input_resolution_yolov3_HW = (800, 800)

    runner = OnnxTestRunner(args.onnx, args.input_img, input_resolution_yolov3_HW, args.output_img, args.train_csv)
    runner.test()

if __name__ == '__main__':
    main()
