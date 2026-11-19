"""
Training augmentation and batch collation, copied VERBATIM from the
pre-revision data.py (SegmentationAugmentor, collate_fn). Do not edit the
bodies: the training data stream must stay identical to the reported runs.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A

from .utils import sample_points_safe


class SegmentationAugmentor:
    """
    Segmentation augmentor aligned with SAM preprocessing:
    - Resize longest side to 1024
    - Pad to 1024x1024
    - Light photometric + geometric augmentation

    When oracle prompt coordinates are present in the sample dict
    (keys: 'oracle_bbox', 'oracle_pos_points', 'oracle_neg_points'),
    the geometric transform is applied to them as well via albumentations'
    native keypoints and bboxes support.
    """

    def __init__(self, target_size=1024):
        self.target_size = target_size

        # Photometric: image only, no coordinate impact
        self.photometric = A.Compose([
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.05, p=0.7,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.2),
        ])

        # Geometric: image + mask + keypoints + bboxes
        # keypoint_params: (x, y) format, remove_invisible=True drops
        # points that land outside the image after the transform.
        # bbox_params: pascal_voc format (x_min, y_min, x_max, y_max).
        self.geometric = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.03,
                    scale_limit=0.15,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                    mask_value=0,
                    p=0.5,
                ),
            ],
            keypoint_params=A.KeypointParams(
                format='xy',
                remove_invisible=True,
                angle_in_degrees=True,
            ),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['bbox_labels'],
                min_visibility=0.1,
            ),
        )

    def _resize_and_pad_sam_style(self, image, mask):
        h, w = image.shape[:2]
        target = self.target_size
        if h >= w:
            scale = target / h
            new_h, new_w = target, int(round(w * scale))
        else:
            scale = target / w
            new_h, new_w = int(round(h * scale)), target

        image_resized = cv2.resize(image, (new_w, new_h),
                                   interpolation=cv2.INTER_LINEAR)
        mask_resized  = cv2.resize(mask.astype(np.uint8), (new_w, new_h),
                                   interpolation=cv2.INTER_NEAREST)

        pad_h = target - new_h
        pad_w = target - new_w
        image_padded = cv2.copyMakeBorder(
            image_resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask_padded  = cv2.copyMakeBorder(
            mask_resized,  0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=0)

        return image_padded, mask_padded.astype(bool), scale, new_h, new_w

    def _transform_points(self, points, aug_result, key):
        """
        Recover transformed (x, y) coordinates for a set of keypoints.
        aug_result['keypoints'] is a flat list of (x, y) for all keypoints
        combined; we use the index range stored in aug_result[key+'_range'].
        Returns np.array (N, 2) or None if no points survive.
        """
        start, end = aug_result[key + '_range']
        pts = aug_result['keypoints'][start:end]
        if len(pts) == 0:
            return None
        return np.array([[p[0], p[1]] for p in pts], dtype=np.float32)

    def _scale_and_pad_points(self, points, scale, new_h, new_w):
        """Apply the same resize+pad as the image to point coordinates."""
        if points is None or len(points) == 0:
            return None
        pts = points.copy().astype(np.float32)
        pts[:, 0] *= scale   # x
        pts[:, 1] *= scale   # y
        # Padding is bottom/right only → no offset needed for x/y
        # Clip to valid padded canvas region
        pts[:, 0] = np.clip(pts[:, 0], 0, new_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, new_h - 1)
        return pts

    def _scale_and_pad_bbox(self, bbox, scale, new_h, new_w, target):
        """Apply resize+pad to a bbox [x_min, y_min, x_max, y_max]."""
        if bbox is None:
            return None
        x_min, y_min, x_max, y_max = bbox
        x_min = float(np.clip(x_min * scale, 0, new_w - 1))
        y_min = float(np.clip(y_min * scale, 0, new_h - 1))
        x_max = float(np.clip(x_max * scale, 0, new_w - 1))
        y_max = float(np.clip(y_max * scale, 0, new_h - 1))
        return [x_min, y_min, x_max, y_max]

    def __call__(self, sample):
        """
        sample keys (required):
            image    : np.ndarray (H, W, 3)
            gt_mask  : np.ndarray (H, W) {0,1} or bool

        sample keys (optional, oracle prompts):
            oracle_pos_points : np.ndarray (N, 2) [x, y] or None
            oracle_neg_points : np.ndarray (N, 2) [x, y] or None
            oracle_bbox       : list [x_min, y_min, x_max, y_max] or None
        """
        image = sample['image']
        mask  = sample['gt_mask'].astype(np.uint8)

        # Oracle prompt fields (may be absent for non-distillation training)
        pos_pts  = sample.get('oracle_pos_points', None)
        neg_pts  = sample.get('oracle_neg_points', None)
        bbox     = sample.get('oracle_bbox', None)

        has_oracle = (pos_pts is not None or neg_pts is not None
                      or bbox is not None)

        # ── Photometric (image only) ─────────────────────────────────────────
        image = self.photometric(image=image)['image']

        # ── Geometric ────────────────────────────────────────────────────────
        if has_oracle:
            # Pack all keypoints into one flat list, remembering index ranges
            # so we can recover pos/neg separately after the transform.
            all_kpts = []
            pos_range = (0, 0)
            neg_range = (0, 0)

            if pos_pts is not None and len(pos_pts) > 0:
                pos_range = (len(all_kpts), len(all_kpts) + len(pos_pts))
                all_kpts.extend([(float(p[0]), float(p[1])) for p in pos_pts])

            if neg_pts is not None and len(neg_pts) > 0:
                neg_range = (len(all_kpts), len(all_kpts) + len(neg_pts))
                all_kpts.extend([(float(p[0]), float(p[1])) for p in neg_pts])

            # Pack bbox
            bboxes_in = []
            bbox_labels = []
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                # albumentations requires x_max > x_min and y_max > y_min
                if x_max > x_min and y_max > y_min:
                    bboxes_in    = [[x_min, y_min, x_max, y_max]]
                    bbox_labels  = ['oracle']

            aug = self.geometric(
                image=image, mask=mask,
                keypoints=all_kpts,
                bboxes=bboxes_in,
                bbox_labels=bbox_labels,
            )
            image, mask = aug['image'], aug['mask']

            # Recover transformed points
            aug['pos_range'] = pos_range
            aug['neg_range'] = neg_range

            def _recover(start, end):
                pts = aug['keypoints'][start:end]
                if len(pts) == 0:
                    return None
                return np.array([[p[0], p[1]] for p in pts], dtype=np.float32)

            pos_pts = _recover(*pos_range)
            neg_pts = _recover(*neg_range)
            bbox    = aug['bboxes'][0][:4] if len(aug['bboxes']) > 0 else None

        else:
            aug   = self.geometric(image=image, mask=mask,
                                   keypoints=[], bboxes=[], bbox_labels=[])
            image = aug['image']
            mask  = aug['mask']

        # ── Resize + pad SAM-style ────────────────────────────────────────────
        image, mask, scale, new_h, new_w = self._resize_and_pad_sam_style(
            image, mask)

        if has_oracle:
            pos_pts = self._scale_and_pad_points(pos_pts, scale, new_h, new_w)
            neg_pts = self._scale_and_pad_points(neg_pts, scale, new_h, new_w)
            bbox    = self._scale_and_pad_bbox(
                bbox, scale, new_h, new_w, self.target_size)

        sample['image']   = image
        sample['gt_mask'] = mask.astype(bool)

        if has_oracle:
            sample['oracle_pos_points'] = pos_pts
            sample['oracle_neg_points'] = neg_pts
            sample['oracle_bbox']       = bbox

        return sample


def collate_fn(batch):
    batched_input    = []
    batched_gt_masks = []

    for sample in batch:
        image, gt_mask = sample['image'], sample['gt_mask']
        try:
            point_coords = sample_points_safe(gt_mask, sample['K'])
        except Exception:
            continue

        num_pts_per_inst = len(point_coords)
        if not num_pts_per_inst:
            continue

        point_coords = torch.unsqueeze(torch.from_numpy(point_coords), 1)
        point_labels = torch.ones(point_coords.shape[0])[:, None]
        image        = torch.tensor(image.copy(), dtype=torch.float32).permute(2, 0, 1)

        batched_input.append({
            'image':         image.to(sample['device']),
            'original_size': (image.shape[1], image.shape[2]),
            'point_coords':  point_coords.to(sample['device']),
            'point_labels':  point_labels.to(sample['device']),
        })
        gt_mask = torch.unsqueeze(torch.tensor(gt_mask, dtype=torch.float32), 0)
        batched_gt_masks.extend([gt_mask for _ in range(num_pts_per_inst)])

    if not len(batched_gt_masks):
        return [], [] 

    batched_gt_masks = torch.stack(batched_gt_masks).to(sample['device'])
    batched_gt_masks = F.interpolate(
        batched_gt_masks, size=(256, 256),
        mode='bilinear', align_corners=False, antialias=True,
    )
    batched_gt_masks = (batched_gt_masks >= 0.5).float().squeeze(1)
    return batched_input, batched_gt_masks