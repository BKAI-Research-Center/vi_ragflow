#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import copy
import time
import os
from PIL import Image

from huggingface_hub import snapshot_download

from api.utils.file_utils import get_project_base_directory
from .operators import *  # noqa: F403
from . import operators
import math
import numpy as np
import cv2
import onnxruntime as ort
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor

from .postprocess import build_post_process

loaded_models = {}

def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(
        op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = getattr(operators, op_name)(**param)
        ops.append(op)
    return ops


def load_model(model_dir, nm):
    logging.info(f"[INFO] Confirm mounting and load_model {model_dir} {nm}")
    model_file_path = os.path.join(model_dir, nm + ".onnx")
    global loaded_models
    loaded_model = loaded_models.get(model_file_path)
    if loaded_model:
        logging.info(f"load_model {model_file_path} reuses cached model")
        return loaded_model

    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path))

    def cuda_is_available():
        try:
            import torch
            if torch.cuda.is_available():
                return True
        except Exception:
            return False
        return False

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 2

    # https://github.com/microsoft/onnxruntime/issues/9509#issuecomment-951546580
    # Shrink GPU memory after execution
    run_options = ort.RunOptions()
    if cuda_is_available():
        cuda_provider_options = {
            "device_id": 0, # Use specific GPU
            "gpu_mem_limit": 512 * 1024 * 1024, # Limit gpu memory
            "arena_extend_strategy": "kNextPowerOfTwo",  # gpu memory allocation strategy
        }
        sess = ort.InferenceSession(
            model_file_path,
            options=options,
            providers=['CUDAExecutionProvider'],
            provider_options=[cuda_provider_options]
            )
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
        logging.info(f"load_model {model_file_path} uses GPU")
    else:
        sess = ort.InferenceSession(
            model_file_path,
            options=options,
            providers=['CPUExecutionProvider'])
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu")
        logging.info(f"load_model {model_file_path} uses CPU")
    loaded_model = (sess, run_options)
    loaded_models[model_file_path] = loaded_model
    return loaded_model


class OCR(object):
    def __init__(self, model_dir=None):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """
        """
        Initialize the OCR module with Surya OCR predictors.

        Args:
            model_dir (str, optional): Directory path (unused in this implementation, kept for compatibility).
        """
        # Initialize Surya predictors
        self.detection_predictor = DetectionPredictor()
        self.recognition_predictor = RecognitionPredictor()
        self.drop_score = 0.5
        self.crop_image_res_index = 0
        self.langs = ["en", "vi"]

    def get_rotate_crop_image(self, img, points):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                    _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break

        return _boxes

    def detect(self, img):
        """
        Detect text regions in the image using Surya OCR.

        Args:
            img (np.ndarray): Input image as a NumPy array.

        Returns:
            zip: Iterator of (sorted_boxes, dummy_scores) where sorted_boxes are the detected text boxes,
                 and dummy_scores are placeholders (empty string and 0) for compatibility.
        """
        time_dict = {"det": 0, "rec": 0, "cls": 0, "all": 0}
        if img is None:
            return None, None, time_dict

        start = time.time()
        # Convert NumPy array to PIL Image for Surya
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Use Surya detection predictor
        predictions = self.detection_predictor([image])
        # Assuming predictions[0]['boxes'] returns boxes in [x1, y1, x2, y2] format, convert to [4, 2]
        dt_boxes = np.array(
            [box.polygon for box in predictions[0].bboxes], dtype=np.float32
        )
        dt_boxes = self.sorted_boxes(dt_boxes)
        elapse = time.time() - start
        time_dict["det"] = elapse
        time_dict["all"] = elapse

        if len(dt_boxes) == 0:
            return None, None, time_dict

        # Return boxes with dummy scores for compatibility
        return zip(dt_boxes, [("", 0) for _ in range(len(dt_boxes))])

    def recognize(self, ori_im, box):
        """
        Recognize text in a cropped image region using Surya OCR.

        Args:
            ori_im (np.ndarray): Original image as a NumPy array.
            box (np.ndarray): Bounding box coordinates with shape [4, 2].

        Returns:
            str: Recognized text if confidence >= drop_score, else empty string.
        """
        img_crop = self.get_rotate_crop_image(ori_im, box)
        # Convert cropped image to PIL Image for Surya
        image = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
        img_width, img_height = image.size
        # Use Surya recognition predictor (no language specified, as recommended)
        predictions = self.recognition_predictor(
            images=[image],
            langs=[self.langs],
            polygons=[
                [
                    [
                        [0, 0],  # Top-left corner
                        [img_width, 0],  # Top-right corner
                        [img_width, img_height],  # Bottom-right corner
                        [0, img_height],  # Bottom-left corner
                    ]
                ]
            ],
        )
        # Assuming predictions return a list of dicts with 'text' and 'score' keys
        text, score = (
            predictions[0].text_lines[0].text,
            predictions[0].text_lines[0].confidence,
        )
        if score < self.drop_score:
            return ""
        return text + " "

    def __call__(self, img, cls=True):
        """
        Perform OCR on the image using Surya OCR.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            cls (bool): Unused parameter, kept for compatibility.

        Returns:
            list: List of (box, (text, score)) tuples for recognized text regions.
            None: Additional return value for compatibility.
            dict: Timing dictionary with 'det', 'rec', 'cls', and 'all' keys.
        """
        time_dict = {"det": 0, "rec": 0, "cls": 0, "all": 0}
        if img is None:
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        # Convert NumPy array to PIL Image for Surya
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Use Surya detection predictor
        det_predictions = self.detection_predictor([image])
        # Convert boxes to [4, 2] format
        dt_boxes = np.array(
            [box.polygon for box in det_predictions[0].bboxes], dtype=np.float32
        )
        time_dict["det"] = time.time() - start

        if dt_boxes is None:
            end = time.time()
            time_dict["all"] = end - start
            return None, None, time_dict

        # Prepare cropped images for recognition
        img_crop_list = []
        dt_boxes = self.sorted_boxes(dt_boxes)
        crop_dt_boxes_polygons = []

        for bno in range(len(dt_boxes)):
            tmp_box = np.array(copy.deepcopy(dt_boxes[bno]))
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(
                Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
            )
            crop_dt_boxes_polygons.append([tmp_box - tmp_box[0]])

        # Use Surya recognition predictor on cropped images
        rec_predictions = self.recognition_predictor(
            images=img_crop_list,
            langs=[self.langs] * len(img_crop_list),
            polygons=np.array(crop_dt_boxes_polygons, dtype=np.int32),
        )
        rec_res = [
            (pred.text_lines[0].text + " ", pred.text_lines[0].confidence)
            for pred in rec_predictions
        ]
        time_dict["rec"] = time.time() - start - time_dict["det"]

        # Filter results based on drop_score
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        time_dict["all"] = time.time() - start

        # Convert boxes to list format for compatibility
        return list(zip([box.tolist() for box in filter_boxes], filter_rec_res))
