import os
import pickle
from pathlib import Path

import numpy as np
import tqdm
from ultralytics import YOLO

from tools.log import logger
from tools.video_source import get_video_gen


class Evaluation:

    def __init__(self,
                 src_path,
                 read_start=0,
                 total_num=100,
                 resize=[720, 1280],
                 gt_dir='../ground_truth',
                 gt_ckpt_path='~/files/VideoAnalyticsFramework/models/yolov8n.pt'):
        """Evaluation for object detection. Detection results in format of nx1,ny1,nx2,ny2,conf,cls"""
        self.src_path = src_path
        self.read_start = read_start
        self.total_num = total_num
        self.resize = resize
        self.gt_dir = gt_dir
        self.gt_ckpt_path = gt_ckpt_path

        # get video name
        self.video_name = Path(src_path).name

        # load existing ground truth results if available, otherwise generate
        self.load_gt()

    def load_gt(self):
        """Load ground truth results."""
        gt_path = os.path.join(self.gt_dir, self.video_name) + '.pkl'

        if not os.path.isfile(gt_path):
            self.generate_gt(gt_path)

        with open(gt_path, 'rb') as f:
            results = pickle.load(f)

        if self.read_start + self.total_num > results['num_frames']:
            results = self.generate_gt(gt_path)
        else:
            logger.info(f'Using existing ground truth results: {gt_path}')

        self.gt_preds = results['preds']

    def get_gt(self):
        return self.gt_preds

    def generate_gt(self, gt_path):
        """Generate ground truth images."""
        model = YOLO(self.gt_ckpt_path)

        if os.path.isfile(gt_path):
            with open(gt_path, 'rb') as f:
                results = pickle.load(f)
            logger.info(
                f'Found local file {gt_path}, checking for consistency...')
            assert results[
                'video_name'] == self.video_name, f"local file video_name {results['video_name']} is different with current video name {self.video_name}"
            assert results['num_frames'] == len(
                results['preds']
            ), f"num_frames {results['num_frames']} is different with available preds {len(results['preds'])}"
            if results['num_frames'] >= self.read_start + self.total_num:
                logger.warning(
                    f'local file num_frames {results["num_frames"]} >= current num_frames {self.num_frames}, why generate again?'
                )
                return
            assert results[
                'resize'] == self.resize, f"local file resize {results['resize']} is different with current resize {self.resize}"

            existing_num_frames = results['num_frames']
            results['num_frames'] = self.read_start + self.total_num
            remaining_num_frames = self.read_start + self.total_num - existing_num_frames
            logger.info(
                f'Generating {remaining_num_frames} remaining results from idx {existing_num_frames}'
            )

            self.video_gen = get_video_gen(src_path=self.src_path,
                                           resize=self.resize,
                                           fps=None,
                                           read_start=existing_num_frames,
                                           max_num=remaining_num_frames,
                                           batch_size=1)

            for idx, frame in tqdm.tqdm(self.video_gen.read()):
                preds = model.predict(frame, verbose=False)
                boxes = np.array(preds[0].boxes.xyxyn.cpu())
                conf = np.array(preds[0].boxes.conf.cpu())
                classes = np.array(preds[0].boxes.cls.cpu())
                results['preds'][idx] = np.concatenate(
                    (boxes, conf[:, None], classes[:, None]), axis=1)
        else:
            logger.info(
                f'Generating ground truth results: {gt_path} with model: {self.gt_ckpt_path}'
            )
            os.makedirs(self.gt_dir, exist_ok=True)

            results = {}
            results['src_path'] = self.src_path
            results['video_name'] = self.video_name
            results['num_frames'] = self.read_start + self.total_num
            results['resize'] = self.resize
            results['preds'] = {}

            self.video_gen = get_video_gen(src_path=self.src_path,
                                           resize=self.resize,
                                           fps=None,
                                           read_start=0,
                                           max_num=self.read_start +
                                           self.total_num,
                                           batch_size=1)

            for idx, frame in tqdm.tqdm(self.video_gen.read()):
                preds = model.predict(frame, verbose=False)
                boxes = np.array(preds[0].boxes.xyxyn.cpu())
                conf = np.array(preds[0].boxes.conf.cpu())
                classes = np.array(preds[0].boxes.cls.cpu())
                results['preds'][idx] = np.concatenate(
                    (boxes, conf[:, None], classes[:, None]), axis=1)

        # Save results
        with open(gt_path, 'wb') as f:
            pickle.dump(results, f)

        logger.info(f'gt saved to {gt_path}')
        return results

    def eval(self, idx, preds, gt=None):
        """Evaluate the prediction."""
        if gt is None:
            gt = self.gt_preds.get(idx, [])

        # keep only two classes, 0: person, 1: vehicle
        gt = gt[gt[:, -1] < 8]
        preds = preds[preds[:, -1] < 8]
        gt[gt[:, -1] != 0, -1] = 1
        preds[preds[:, -1] != 0, -1] = 1

        # remove box in preds that are too large
        preds = preds[(preds[:, 2] - preds[:, 0]) *
                      (preds[:, 3] - preds[:, 1]) < 0.5]

        # gt info
        if len(gt) > 0:
            mean_obj_size = np.mean(
                (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))
        else:
            mean_obj_size = 0
        obj_num = len(gt)

        return self.calculate_map50(preds, gt), mean_obj_size, obj_num

    def calculate_map50(self, preds, gt):
        """Calculate mAP50.

        Args:
            preds: (N, 6) array of predictions, [x1, y1, x2, y2, conf, class_id]
            gt: (N, 6) array of ground truth, [x1, y1, x2, y2, conf, class_id]
            all the boxes are in normalized coordinates.

        Returns:
            map50: mAP50
            tp_box: (N, 6) list of true positive boxes, green
            fp_box: (N, 6) list of false positive boxes, yellow
            fn_box: (N, 6) list of false negative boxes, red
        """
        if len(preds) == 0 and len(gt) == 0:
            return 1, [], [], []

        uniq_classes = []
        if len(gt) > 0:
            uniq_classes += np.unique(gt[:, 5]).tolist()
        if len(preds) > 0:
            uniq_classes += np.unique(preds[:, 5]).tolist()
        uniq_classes = list(set(uniq_classes))
        average_precisions = []
        tp_box, fp_box, fn_box = [], [], []

        for class_id in uniq_classes:
            # Filter predictions and ground truth for the current class
            class_preds = preds[preds[:, 5] ==
                                class_id] if len(preds) > 0 else []
            class_gt = gt[gt[:, 5] == class_id] if len(gt) > 0 else []

            if len(class_preds) == 0:
                if len(class_gt) != 0:
                    average_precisions.append(0)
                    fn_box += class_gt.tolist()
                    continue
            else:
                if len(class_gt) == 0:
                    average_precisions.append(0)
                    fp_box += class_preds.tolist()
                    continue

            # Sort predictions by confidence (descending order)
            sorted_preds = class_preds[np.argsort(class_preds[:, 4])[::-1]]

            true_positives = np.zeros(len(sorted_preds))
            false_positives = np.zeros(len(sorted_preds))
            num_gt_boxes = len(class_gt)

            for i, pred in enumerate(sorted_preds):
                overlaps = []
                for j, gt_box in enumerate(class_gt):
                    iou = self.calculate_iou(pred[:4], gt_box[:4])
                    overlaps.append((iou, j))

                    if iou > 0.5:
                        class_gt[j, 4] = -1

                # Sort by IoU in descending order
                overlaps = sorted(overlaps, key=lambda x: x[0], reverse=True)
                best_overlap = overlaps[0]

                if best_overlap[0] >= 0.5:
                    true_positives[i] = 1
                    tp_box.append(pred.tolist())
                else:
                    false_positives[i] = 1
                    fp_box.append(pred.tolist())

            # Compute precision and recall
            cum_true_positives = np.cumsum(true_positives)
            cum_false_positives = np.cumsum(false_positives)
            recall = cum_true_positives / num_gt_boxes
            precision = cum_true_positives / (cum_true_positives +
                                              cum_false_positives)

            # Compute average precision
            recall_levels = np.linspace(0, 1, 11)
            interpolated_precision = []
            for target_recall in recall_levels:
                precision_at_recall_level = max(
                    precision[recall >= target_recall]) if len(
                        precision[recall >= target_recall]) > 0 else 0
                interpolated_precision.append(precision_at_recall_level)
            ap = np.mean(interpolated_precision)

            average_precisions.append(ap)

            # Append false negative boxes
            fn_box += class_gt[class_gt[:, 4] != -1].tolist()

        # Calculate mAP50 by averaging the APs for all classes
        if len(average_precisions) == 0:
            map50 = 0
        else:
            map50 = sum(average_precisions) / len(average_precisions)

        assert len(tp_box) + len(fp_box) == len(
            preds
        ), f'check failed len(tp_box) ({len(tp_box)}) + len(fp_box) ({len(fp_box)}) != len(preds) ({len(preds)})'
        return map50, tp_box, fp_box, fn_box

    def calculate_iou(self, box1, box2):
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        # Calculate IoU
        iou = intersection / union
        return iou
