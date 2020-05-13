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
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw, Image
from pyFormulaClientNoNvidia import messages
import argparse
from timeit import default_timer as timer
import csv

import wget
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os

class TestRunner:
    def __init__(self, input_file_path, input_resolution, out_image, train_csv):
        self.input_file_path = input_file_path
        camera_data = TestRunner.get_camera_data(input_file_path)
        self.raw_image = Image.frombytes("RGB", (camera_data.width, camera_data.height), camera_data.pixels, 'raw', 'RGBX', 0,-1)
        self.input_resolution = input_resolution
        self.out_image = out_image

        self.output_shapes = [(1, 255, 25, 25), (1, 255, 50, 50), (1, 255, 100, 100)]
        vanilla_anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), 
                            (59, 119), (116, 90), (156, 198), (373, 326)]
        if train_csv is not None:
            with open(train_csv) as f:
                csv_reader = csv.reader(f)
                row = next(csv_reader)
                row = str(row)[2:-2]
                self.anchors = [tuple([float(y) for y in x.split(',')]) for x in row.split("'")[0].split('|')]
        else:
            self.anchors = vanilla_anchors

    def _preprocess(self):
        start = timer()
        # Create a pre-processor object by specifying the required input resolution for YOLOv3
        preprocessor = PreprocessYOLO(self.input_resolution)
        # Load an image from the specified input path, and return it together with  a pre-processed version
        return preprocessor.process(self.raw_image)

    def _post_process(self, output):
        # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
        output = [output.reshape(shape) for output, shape in zip(output, self.output_shapes)]

        postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                            "yolo_anchors": self.anchors,                                          # A list of 9 two-dimensional tuples for the YOLO anchors
                            "obj_threshold": 0.5,                                               # Threshold for object coverage, float value between 0 and 1
                            "nms_threshold": 0.25,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                            "yolo_input_resolution": self.input_resolution}

        postprocessor = PostprocessYOLO(**postprocessor_args)

        # # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
        return postprocessor.process(output, (self.raw_image.size))

    def _save_out_image(self, boxes, scores, classes):
        # # Draw the bounding boxes onto the original input image and save it as a PNG file
        obj_detected_img = TestRunner.draw_bboxes(self.raw_image, boxes, scores, classes, ALL_CATEGORIES)
        obj_detected_img.save(self.out_image, 'PNG')

    def _initialize(self):
        raise NotImplementedError()

    def _infer(self, input_data):
        raise NotImplementedError()

    def _destroy(self):
        pass

    def test(self):
        print("Initializng")
        start = timer()
        self._initialize()
        initialize_time = timer() - start

        print("Preprocessing")
        start = timer()
        input_data = self._preprocess()
        preprocess_time = timer() - start

        print('Running inference on image {}...'.format(self.input_file_path))
        start = timer()
        output = self._infer(input_data)
        infer_time = timer() - start

        print("Post processing")
        start = timer()
        boxes, classes, scores = self._post_process(output)
        postprocess_time = timer() - start

        self._save_out_image(boxes, scores, classes)
        print('Saved image with bounding boxes of detected objects to {}.'.format(self.out_image))

        self._destroy()

        image_processing_time = preprocess_time + infer_time + postprocess_time

        print(f"Initializing took {round((initialize_time)*1000)} [ms] ")
        print(f"Preprocessing took {round((preprocess_time)*1000)} [ms] ")
        print(f"Inference took {round((infer_time)*1000)} [ms] ")
        print(f"Postprocessing took {round((postprocess_time)*1000)} [ms] ")
        print(f"Overall processing image (without initializing the network) took {round((image_processing_time)*1000)} [ms] ")

    @staticmethod
    def get_camera_data(message_path):
        with open(message_path, 'rb') as f:
            buffer = f.read()
            camera_msg = messages.common.Message()
            camera_msg.ParseFromString(buffer)
        camera_data = messages.sensors.CameraSensor()
        camera_msg.data.Unpack(camera_data)
        return camera_data

    @staticmethod
    def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
        """Draw the bounding boxes on the original input image and return it.

        Keyword arguments:
        image_raw -- a raw PIL Image
        bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
        categories -- NumPy array containing the corresponding category for each object,
        with shape (N,)
        confidences -- NumPy array containing the corresponding confidence for each object,
        with shape (N,)
        all_categories -- a list of all categories in the correct ordered (required for looking up
        the category name)
        bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
        """
        draw = ImageDraw.Draw(image_raw)
        print(bboxes, confidences, categories)
        for box, score, category in zip(bboxes, confidences, categories):
            x_coord, y_coord, width, height = box
            left = max(0, np.floor(x_coord + 0.5).astype(int))
            top = max(0, np.floor(y_coord + 0.5).astype(int))
            right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
            bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

            draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
            draw.text((left, top - 12), '{:.2f}'.format(score), fill=bbox_color)

        return image_raw

