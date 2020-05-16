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
import torch
import torchvision
from utils.nms import nms
from utils.utils import calculate_padding

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
        self.image_height = self.input_resolution[0]
        self.image_width = self.input_resolution[1]
        self.num_classes = 80 # Should read from file
        self.bbox_attrs = self.num_classes + 5
        self.conf_thres = 0.5
        self.nms_thres = 0.25

        self.yolo_masks = [(6, 7, 8), (3, 4, 5), (0, 1, 2)]
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

    def _forward_yolo_output(self, sample, anchors):
        nA = len(anchors)
        nB = sample.size(0)
        nGh = sample.size(2)
        nGw = sample.size(3)
        stride = self.image_height / nGh
        print(sample.size(), self.bbox_attrs, nB, nA, nGh, nGw)
        prediction = sample.view(nB, nA, self.bbox_attrs, nGh, nGw).permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nGw, dtype=torch.float, device=x.device).repeat(nGh, 1).view([1, 1, nGh, nGw])
        grid_y = torch.arange(nGh, dtype=torch.float, device=x.device).repeat(nGw, 1).t().view([1, 1, nGh, nGw]).contiguous()
        scaled_anchors = torch.tensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors], dtype=torch.float, device=x.device)
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = torch.zeros(prediction[..., :4].shape, dtype=torch.float, device=x.device)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        print(pred_boxes.size(), pred_conf.size(), pred_cls.size())
        # If not in training phase return predictions
        output = torch.cat((
                pred_boxes.view(nB, -1, 4) * stride,
                pred_conf.view(nB, -1, 1),
                pred_cls.view(nB, -1, self.num_classes)),
                -1)
        return output

    @staticmethod
    def mem_to_tensor(alloc, shape):
        '''Convert a :class:`pycuda.gpuarray.GPUArray` to a :class:`torch.Tensor`. The underlying
        storage will NOT be shared, since a new copy must be allocated.
        Parameters
        ----------
        gpuarray  :   pycuda.gpuarray.GPUArray
        Returns
        -------
        torch.Tensor
        '''
        out = torch.zeros(shape, dtype=torch.float32).cuda()
        pycuda.driver.memcpy_dtod(out.data_ptr(), alloc.device, alloc.host.nbytes)
        return out

    def _post_process(self, output):
        # start = timer()
        # A = TestRunner.mem_to_tensor(output[0], self.output_shapes[0])
        # B = TestRunner.mem_to_tensor(output[1], self.output_shapes[1])
        # C = TestRunner.mem_to_tensor(output[2], self.output_shapes[2])
        # tensor_creation_time = timer() - start
        # print(f"Creating tensor {round((tensor_creation_time)*1000)} [ms] ")
        # print(A.size(), B.size(), C.size())
        # anchors_A = ([self.anchors[i] for i in self.yolo_masks[0]])
        # anchors_B = ([self.anchors[i] for i in self.yolo_masks[1]])
        # anchors_C = ([self.anchors[i] for i in self.yolo_masks[2]])
        # print(anchors_A, anchors_B, anchors_C)
        # output_A = self._forward_yolo_output(A, anchors_A)
        # output_B = self._forward_yolo_output(B, anchors_B)
        # output_C = self._forward_yolo_output(C, anchors_C)
        # print(output_A.size(), output_B.size(), output_C.size())
        # full_output = torch.cat((output_A, output_B, output_C), 1)
        # print(full_output.size())
        # w, h = self.raw_image.size
        # pad_h, pad_w, ratio = calculate_padding(h, w, self.image_height, self.image_width)
        # for detections in full_output:
        #     detections = detections[detections[:, 4] > self.conf_thres]
        #     box_corner = torch.zeros((detections.shape[0], 4), device=detections.device)
        #     xy = detections[:, 0:2]
        #     wh = detections[:, 2:4] / 2
        #     box_corner[:, 0:2] = xy - wh
        #     box_corner[:, 2:4] = xy + wh
        #     probabilities = detections[:, 4]
        #     nms_indices = nms(box_corner, probabilities, self.nms_thres)
        #     main_box_corner = box_corner[nms_indices]
        #     probabilities_nms = probabilities[nms_indices]
        #     if nms_indices.shape[0] == 0:  
        #         continue

        # BB_list = []
        # for i in range(len(main_box_corner)):
        #     x0 = main_box_corner[i, 0].to('cpu').item() / ratio - pad_w
        #     y0 = main_box_corner[i, 1].to('cpu').item() / ratio - pad_h
        #     x1 = main_box_corner[i, 2].to('cpu').item() / ratio - pad_w
        #     y1 = main_box_corner[i, 3].to('cpu').item() / ratio - pad_h 
        #     # draw.rectangle((x0, y0, x1, y1), outline="red")
        #     # print("BB ", i, "| x = ", x0, "y = ", y0, "w = ", x1 - x0, "h = ", y1 - y0, "probability = ", probabilities_nms[i].item())
        #     BB = [round(x0), round(y0), round(y1 - y0), round(x1 - x0)]  # x, y, h, w
        #     BB_list.append(BB)
        # return BB_list, probabilities_nms

        # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
        output = [output.reshape(shape) for output, shape in zip(output, self.output_shapes)]
        

        postprocessor_args = {"yolo_masks": self.yolo_masks,                    # A list of 3 three-dimensional tuples for the YOLO masks
                            "yolo_anchors": self.anchors,                                          # A list of 9 two-dimensional tuples for the YOLO anchors
                            "obj_threshold": 0.5,                                               # Threshold for object coverage, float value between 0 and 1
                            "nms_threshold": 0.25,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                            "yolo_input_resolution": self.input_resolution}

        postprocessor = PostprocessYOLO(**postprocessor_args)

        # # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
        return postprocessor.process(output, (self.raw_image.size))

    def _save_out_image(self, boxes, scores):
        # # Draw the bounding boxes onto the original input image and save it as a PNG file
        obj_detected_img = TestRunner.draw_bboxes(self.raw_image, boxes, scores)
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
        boxes, scores = self._post_process(output)
        postprocess_time = timer() - start

        self._save_out_image(boxes, scores)
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
    def draw_bboxes(image_raw, bboxes, confidences, bbox_color='blue'):
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
        # print(bboxes, confidences)
        for box, score in zip(bboxes, confidences):
            x_coord, y_coord, width, height = box
            left = max(0, np.floor(x_coord + 0.5).astype(int))
            top = max(0, np.floor(y_coord + 0.5).astype(int))
            right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
            bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

            draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
            draw.text((left, top - 12), '{:.2f}'.format(score), fill=bbox_color)

        return image_raw

