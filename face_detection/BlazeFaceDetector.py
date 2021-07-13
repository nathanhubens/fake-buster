import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from face_detection.video_utils import nms


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class BlazeFace(nn.Module):
    """The BlazeFace face detection model from MediaPipe.

    The version from MediaPipe is simpler than the one in the paper;
    it does not use the "double" BlazeBlocks.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv
    weights by TFLite.

    The conversion to PyTorch is fairly straightforward, but there are
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.

    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/google/mediapipe/
    """
    input_size = (128, 128)

    def __init__(self):
        super(BlazeFace, self).__init__()

        # These are the settings from the MediaPipe example graph
        # mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.h_scale = 128.0
        self.w_scale = 128.0
        self.min_score_thresh = 0.75
        self.min_suppression_threshold = 0.3

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            BlazeBlock(24, 24),
            BlazeBlock(24, 28),
            BlazeBlock(28, 32, stride=2),
            BlazeBlock(32, 36),
            BlazeBlock(36, 42),
            BlazeBlock(42, 48, stride=2),
            BlazeBlock(48, 56),
            BlazeBlock(56, 64),
            BlazeBlock(64, 72),
            BlazeBlock(72, 80),
            BlazeBlock(80, 88),
        )

        self.backbone2 = nn.Sequential(
            BlazeBlock(88, 96, stride=2),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
        )

        self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
        self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

        self.regressor_8 = nn.Conv2d(88, 32, 1, bias=True)
        self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)

    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)

        b = x.shape[0]  # batch size, needed for reshaping later

        x = self.backbone1(x)  # (b, 88, 16, 16)
        h = self.backbone2(x)  # (b, 96, 8, 8)

        # Note: Because PyTorch is NCHW but TFLite is NHWC, we need to
        # permute the output from the conv layers before reshaping it.

        c1 = self.classifier_8(x)  # (b, 2, 16, 16)
        c1 = c1.permute(0, 2, 3, 1)  # (b, 16, 16, 2)
        c1 = c1.reshape(b, -1, 1)  # (b, 512, 1)

        c2 = self.classifier_16(h)  # (b, 6, 8, 8)
        c2 = c2.permute(0, 2, 3, 1)  # (b, 8, 8, 6)
        c2 = c2.reshape(b, -1, 1)  # (b, 384, 1)

        c = torch.cat((c1, c2), dim=1)  # (b, 896, 1)

        r1 = self.regressor_8(x)  # (b, 32, 16, 16)
        r1 = r1.permute(0, 2, 3, 1)  # (b, 16, 16, 32)
        r1 = r1.reshape(b, -1, 16)  # (b, 512, 16)

        r2 = self.regressor_16(h)  # (b, 96, 8, 8)
        r2 = r2.permute(0, 2, 3, 1)  # (b, 8, 8, 96)
        r2 = r2.reshape(b, -1, 16)  # (b, 384, 16)

        r = torch.cat((r1, r2), dim=1)  # (b, 896, 16)
        return [r, c]

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.classifier_8.weight.device

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_anchors(self, path):
        self.anchors = torch.tensor(np.load(path), dtype=torch.float32, device=self._device())
        assert (self.anchors.ndimension() == 2)
        assert (self.anchors.shape[0] == self.num_anchors)
        assert (self.anchors.shape[1] == 4)

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.float() / 127.5 - 1.0

    def predict_on_image(self, img):
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be
                 128 pixels.

        Returns:
            A tensor with face detections.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))

        return self.predict_on_batch(img.unsqueeze(0))[0]

    def predict_on_batch(self, x, apply_nms=True):
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.
            apply_nms: pass False to not apply non-max suppression

        Returns:
            A list containing a tensor of face detections for each image in
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).

        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        assert x.shape[1] == 3
        assert x.shape[2] == 128
        assert x.shape[3] == 128

        # 1. Preprocess the images into tensors:
        x = x.to(self._device())
        x = self._preprocess(x)

        # 2. Run the neural network:
        with torch.no_grad():
            out = self.__call__(x)

        # 3. Postprocess the raw predictions:
        detections = self._tensors_to_detections(out[0], out[1], self.anchors)

        # 4. Non-maximum suppression to remove overlapping detections:
        return self.nms(detections) if apply_nms else detections

    def nms(self, detections):
        """Filters out overlapping detections."""
        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(detections[i])
            faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, 17), device=self._device())
            filtered_detections.append(faces)

        return filtered_detections

    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        """The output of the neural network is a tensor of shape (b, 896, 16)
        containing the bounding box regressor predictions, as well as a tensor
        of shape (b, 896, 1) with the classification confidences.

        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 17) tensors, one for each image in
        the batch.

        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """
        assert raw_box_tensor.ndimension() == 3
        assert raw_box_tensor.shape[1] == self.num_anchors
        assert raw_box_tensor.shape[2] == self.num_coords

        assert raw_score_tensor.ndimension() == 3
        assert raw_score_tensor.shape[1] == self.num_anchors
        assert raw_score_tensor.shape[2] == self.num_classes

        assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]

        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)

        thresh = self.score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
        detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)

        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        mask = detection_scores >= self.min_score_thresh

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
            output_detections.append(torch.cat((boxes, scores), dim=-1))

        return output_detections

    def _decode_boxes(self, raw_boxes, anchors):
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        boxes = torch.zeros_like(raw_boxes)

        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(6):
            offset = 4 + k * 2
            keypoint_x = raw_boxes[..., offset] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def _weighted_non_max_suppression(self, detections):
        """The alternative NMS method as mentioned in the BlazeFace paper:

        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."

        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.

        The input detections should be a Tensor of shape (count, 17).

        Returns a list of PyTorch tensors, one for each detected face.

        This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto
        """
        if len(detections) == 0: return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        remaining = torch.argsort(detections[:, 16], descending=True)

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = detection.clone()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :16]
                scores = detections[overlapping, 16:17]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(dim=0) / total_score
                weighted_detection[:16] = weighted
                weighted_detection[16] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections

    # IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(box.unsqueeze(0), other_boxes).squeeze(0)


class BlazeFaceDetector:

    def __init__(self, weights="blazeface.pth", anchors="anchors.npy", device=None):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detector = BlazeFace().to(device)

        self.detector.load_weights(weights)
        self.detector.load_anchors(anchors)

    def detect_on_multiple_frames(self, frames):

        resized_frames = np.zeros((len(frames), 128, 128, 3))

        # Resize all frames to 128x128
        for i, frame in enumerate(frames):
            resized_frames[i] = cv2.resize(frame, (128, 128))

        batch_detections = self.detector.predict_on_batch(resized_frames)
        all_formatted_detections = []

        for frame, detections in zip(frames, batch_detections):
            formatted_detections = BlazeFaceDetector._format_detections(detections, frame.shape[0], frame.shape[1])

            # NOTE: Only add the detections if there are any
            all_formatted_detections.append(formatted_detections)

        return all_formatted_detections

    def detect(self, frame):

        # The detections we get back from BlazeFace have two problems:
        # 1. They're formatted differently than what we'd like
        # 2. They were made on a square image, which distorts the bounding box
        detections = self.detector.predict_on_image(cv2.resize(frame, (128, 128)))

        formatted_detections = BlazeFaceDetector._format_detections(detections, frame.shape[0], frame.shape[1])
        return np.array(formatted_detections)

    @staticmethod
    def _format_detections(detections, frame_height, frame_width, num_frames=1):

        formatted_detections = []

        for detection in detections:
            y_min, x_min, y_max, x_max, *_ = detection
            probability = detection[-1]

            # We need, center height and width
            height_ratio = ((frame_height / frame_width) + 1) / 2
            width_ratio = ((frame_width / frame_height) + 1) / 2

            center_y = (y_max + y_min) / 2
            center_x = (x_min + x_max) / 2

            width = x_max - x_min
            height = y_max - y_min

            height = (1 / height_ratio) * height   # undo the original height change
            width = (1 / width_ratio) * width      # undo the original width change

            height = height * 1.2                   # increase height by 20%
            width = width * 1.2                     # increase width by 20%

            # Center is the same, height and width are adjusted by the ratio
            y_min = center_y - (height / 2)
            y_max = center_y + (height / 2)

            x_min = center_x - (width / 2)
            x_max = center_x + (width / 2)

            y_min = torch.ceil(y_min * frame_height)
            x_min = torch.ceil(x_min * frame_width)
            y_max = torch.ceil(y_max * frame_height)
            x_max = torch.ceil(x_max * frame_width)

            # Correct formatting order
            formatted_detections.append([x_min, y_min, x_max, y_max, probability])

        return np.array(formatted_detections)

    @staticmethod
    def _format_detections2(detections, frame_height, frame_width, num_frames=1):

        formatted_detections = []

        for detection in detections:
            y_min, x_min, y_max, x_max, *_ = detection
            probability = detection[-1]

            # Correct formatting order
            formatted_detections.append([x_min, y_min, x_max, y_max, probability])

        return np.array(formatted_detections)

    def get_detections_with_multiple_crops(self, frames):

        if len(frames.shape) < 3 or len(frames.shape) > 4:
            raise Exception("Expected {frames} to have 3 or 4 dimensions. Got: {} dimensions.".format(len(frames.shape)))
        elif len(frames.shape) == 3:
            frames = np.expand_dims(frames, axis=0)

        all_dets = []
        # frames = read_random_frames(video_path, num_frames=num_frames)
        # TODO: Handle errors reading frames
        height, width, _ = frames[0].shape

        if height == width:
            # If it's a square video just predict on it directly
            dets = self.detect_on_multiple_frames(frames)
            return dets

        elif height > width:
            # If it's a vertical video, take three crops
            top = frames[:, :width, :, :]
            center = frames[:, height // 2 - width // 2:height // 2 + width // 2, :, :]
            bottom = frames[:, -width:, :, :]

            top_dets = self.detect_on_multiple_frames(top)
            center_dets = self.detect_on_multiple_frames(center)
            bottom_dets = self.detect_on_multiple_frames(bottom)

            for t_dets, c_dets, b_dets in zip(top_dets, center_dets, bottom_dets):
                frame_dets = []

                # Add top detections for this frame
                for x_min, y_min, x_max, y_max, prob in t_dets:
                    frame_dets.append([float(x_min), float(y_min), float(x_max), float(y_max), float(prob)])

                # Add center detections for this frame
                # Make sure to account for vertical shift
                for x_min, y_min, x_max, y_max, prob in c_dets:
                    offset = height // 2 - width // 2
                    frame_dets.append(
                        [float(x_min), float(y_min + offset), float(x_max), float(y_max + offset), float(prob)])

                # Add bottom detections for this frame
                # Make sure to account for vertical shift
                for x_min, y_min, x_max, y_max, prob in b_dets:
                    offset = height - width
                    frame_dets.append(
                        [float(x_min), float(y_min + offset), float(x_max), float(y_max + offset), float(prob)])

                frame_dets = np.array(frame_dets)
                if len(frame_dets) > 0:
                    inds = nms(frame_dets, 0.5)
                    frame_dets = frame_dets[inds]

                all_dets.append(frame_dets)

        elif height < width:
            # If it's a horizontal video, take three crops
            left = frames[:, :, :height, :]
            center = frames[:, :, width // 2 - height // 2:width // 2 + height // 2, :]
            right = frames[:, :, -height:, :]

            left_dets = self.detect_on_multiple_frames(left)
            center_dets = self.detect_on_multiple_frames(center)
            right_dets = self.detect_on_multiple_frames(right)

            for l_dets, c_dets, r_dets in zip(left_dets, center_dets, right_dets):
                frame_dets = []

                # Add left detections for this frame
                for x_min, y_min, x_max, y_max, prob in l_dets:
                    frame_dets.append([float(x_min), float(y_min), float(x_max), float(y_max), float(prob)])

                # Add center detections for this frame
                # Make sure to account for left shift
                for x_min, y_min, x_max, y_max, prob in c_dets:
                    offset = width // 2 - height // 2
                    frame_dets.append(
                        [float(x_min + offset), float(y_min), float(x_max + offset), float(y_max), float(prob)])

                # Add right detections for this frame
                # Make sure to account for left shift
                for x_min, y_min, x_max, y_max, prob in r_dets:
                    offset = width - height
                    frame_dets.append(
                        [float(x_min + offset), float(y_min), float(x_max + offset), float(y_max), float(prob)])

                frame_dets = np.array(frame_dets)

                if len(frame_dets) > 0:
                    inds = nms(frame_dets, 0.5)
                    frame_dets = frame_dets[inds]

                all_dets.append(frame_dets)

        return all_dets
