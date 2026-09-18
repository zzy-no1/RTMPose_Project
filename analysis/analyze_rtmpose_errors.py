import os
import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ============================================================
# 路径
# ============================================================

PKL_PATH = r"analysis\baseline\rtmpose_m_outputs.pkl"

GT_PATH = (
    r"data\coco\annotations"
    r"\person_keypoints_val2017.json"
)

OUT_DIR = r"analysis\error_analysis"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# COCO 17 个关键点
# ============================================================

JOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


# ============================================================
# bbox 工具
# pred bbox: xyxy
# COCO GT bbox: xywh
# ============================================================

def coco_xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array(
        [x, y, x + w, y + h],
        dtype=np.float32
    )


def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)

    inter = inter_w * inter_h

    area1 = max(
        0.0,
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = max(
        0.0,
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


# ============================================================
# COCO 尺度分类
#
# medium:
# 32^2 <= area < 96^2
#
# large:
# area >= 96^2
# ============================================================

def get_scale_group(area):
    if area < 32 ** 2:
        return "small"
    elif area < 96 ** 2:
        return "medium"
    else:
        return "large"


# ============================================================
# 读取预测结果
# ============================================================

print("Loading RTMPose predictions...")

with open(PKL_PATH, "rb") as f:
    preds = pickle.load(f)

print("Prediction samples:", len(preds))


# 按 image_id 分组
pred_by_image = defaultdict(list)

for item in preds:

    img_id = int(item["img_id"])

    pred_instances = item["pred_instances"]

    pred_kpts = np.asarray(
        pred_instances["keypoints"]
    )[0]

    pred_bbox = np.asarray(
        pred_instances["bboxes"]
    )[0]

    keypoint_scores = np.asarray(
        pred_instances["keypoint_scores"]
    )[0]

    bbox_score = float(
        np.asarray(
            pred_instances["bbox_scores"]
        ).reshape(-1)[0]
    )

    pred_by_image[img_id].append(
        {
            "keypoints": pred_kpts,
            "bbox": pred_bbox,
            "keypoint_scores": keypoint_scores,
            "bbox_score": bbox_score,
        }
    )


# ============================================================
# 读取 COCO GT
# ============================================================

print("Loading COCO ground truth...")

with open(GT_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

gt_by_image = defaultdict(list)

for ann in coco["annotations"]:

    # COCO keypoint annotation
    # num_keypoints = 0 的人没有有效关键点
    if ann.get("num_keypoints", 0) <= 0:
        continue

    img_id = int(ann["image_id"])

    kpts = np.array(
        ann["keypoints"],
        dtype=np.float32
    ).reshape(17, 3)

    gt_by_image[img_id].append(
        {
            "ann_id": int(ann["id"]),
            "keypoints": kpts[:, :2],
            "visible": kpts[:, 2],
            "bbox": coco_xywh_to_xyxy(
                ann["bbox"]
            ),
            "area": float(ann["area"]),
        }
    )


print("GT images:", len(gt_by_image))


# ============================================================
# 匹配参数
# ============================================================

IOU_THRESHOLD = 0.5


# ============================================================
# 统计容器
# ============================================================

joint_errors = defaultdict(list)

joint_errors_medium = defaultdict(list)
joint_errors_large = defaultdict(list)

matched_rows = []

matched_gt = 0
total_gt = 0


# ============================================================
# 按图像做 bbox Hungarian matching
# ============================================================

print("Matching predictions with GT...")

for img_id, gt_list in gt_by_image.items():

    total_gt += len(gt_list)

    pred_list = pred_by_image.get(
        img_id,
        []
    )

    if len(pred_list) == 0:
        continue

    num_gt = len(gt_list)
    num_pred = len(pred_list)

    iou_matrix = np.zeros(
        (num_gt, num_pred),
        dtype=np.float32
    )

    for g, gt in enumerate(gt_list):

        for p, pred in enumerate(pred_list):

            iou_matrix[g, p] = bbox_iou(
                gt["bbox"],
                pred["bbox"]
            )

    # Hungarian：
    # 最大化 IoU
    gt_indices, pred_indices = (
        linear_sum_assignment(
            -iou_matrix
        )
    )

    for g, p in zip(
        gt_indices,
        pred_indices
    ):

        iou = float(
            iou_matrix[g, p]
        )

        if iou < IOU_THRESHOLD:
            continue

        gt = gt_list[g]
        pred = pred_list[p]

        matched_gt += 1

        area = gt["area"]

        scale_group = get_scale_group(
            area
        )

        # 用 sqrt(area) 作为尺度归一化
        # 防止大人物天然产生更大的像素误差
        scale_norm = max(
            np.sqrt(area),
            1.0
        )

        gt_kpts = gt["keypoints"]
        gt_vis = gt["visible"]

        pred_kpts = pred["keypoints"]

        # COCO:
        # v=0 未标注
        # v=1 标注但不可见
        # v=2 可见
        #
        # 这里 v>0 都属于有 GT 标注
        valid = gt_vis > 0

        for j in range(17):

            if not valid[j]:
                continue

            distance = np.linalg.norm(
                pred_kpts[j] -
                gt_kpts[j]
            )

            normalized_error = (
                distance /
                scale_norm
            )

            joint_name = JOINT_NAMES[j]

            joint_errors[
                joint_name
            ].append(
                normalized_error
            )

            if scale_group == "medium":

                joint_errors_medium[
                    joint_name
                ].append(
                    normalized_error
                )

            elif scale_group == "large":

                joint_errors_large[
                    joint_name
                ].append(
                    normalized_error
                )

            matched_rows.append(
                {
                    "image_id": img_id,
                    "ann_id": gt["ann_id"],
                    "joint": joint_name,
                    "scale_group":
                        scale_group,
                    "gt_x":
                        gt_kpts[j][0],
                    "gt_y":
                        gt_kpts[j][1],
                    "pred_x":
                        pred_kpts[j][0],
                    "pred_y":
                        pred_kpts[j][1],
                    "pixel_error":
                        distance,
                    "normalized_error":
                        normalized_error,
                    "bbox_iou":
                        iou,
                    "bbox_score":
                        pred["bbox_score"],
                }
            )


# ============================================================
# Joint-level 汇总
# ============================================================

rows = []

for name in JOINT_NAMES:

    all_errors = np.array(
        joint_errors[name]
    )

    medium_errors = np.array(
        joint_errors_medium[name]
    )

    large_errors = np.array(
        joint_errors_large[name]
    )

    rows.append(
        {
            "joint": name,

            "num_samples":
                len(all_errors),

            "mean_error":
                np.mean(all_errors)
                if len(all_errors)
                else np.nan,

            "median_error":
                np.median(all_errors)
                if len(all_errors)
                else np.nan,

            "medium_mean_error":
                np.mean(medium_errors)
                if len(medium_errors)
                else np.nan,

            "large_mean_error":
                np.mean(large_errors)
                if len(large_errors)
                else np.nan,

            "medium_minus_large":
                (
                    np.mean(medium_errors)
                    -
                    np.mean(large_errors)
                )
                if (
                    len(medium_errors)
                    and len(large_errors)
                )
                else np.nan,
        }
    )


joint_df = pd.DataFrame(rows)

joint_df = joint_df.sort_values(
    "mean_error",
    ascending=False
)

joint_df["rank"] = range(
    1,
    len(joint_df) + 1
)


# ============================================================
# Medium / Large 总体统计
# ============================================================

matched_df = pd.DataFrame(
    matched_rows
)

scale_summary = (
    matched_df[
        matched_df[
            "scale_group"
        ].isin(
            ["medium", "large"]
        )
    ]
    .groupby(
        "scale_group"
    )
    .agg(
        num_joint_samples=(
            "normalized_error",
            "count"
        ),
        mean_error=(
            "normalized_error",
            "mean"
        ),
        median_error=(
            "normalized_error",
            "median"
        ),
        mean_pixel_error=(
            "pixel_error",
            "mean"
        ),
    )
    .reset_index()
)


# ============================================================
# 保存 CSV
# ============================================================

joint_csv = os.path.join(
    OUT_DIR,
    "joint_error_ranking.csv"
)

scale_csv = os.path.join(
    OUT_DIR,
    "scale_error_summary.csv"
)

detail_csv = os.path.join(
    OUT_DIR,
    "matched_joint_details.csv"
)

joint_df.to_csv(
    joint_csv,
    index=False,
    encoding="utf-8-sig"
)

scale_summary.to_csv(
    scale_csv,
    index=False,
    encoding="utf-8-sig"
)

matched_df.to_csv(
    detail_csv,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 控制台打印
# ============================================================

print()
print("=" * 70)
print("MATCHING SUMMARY")
print("=" * 70)

print(
    f"Total GT persons: {total_gt}"
)

print(
    f"Matched GT persons: {matched_gt}"
)

print(
    "GT match rate: "
    f"{matched_gt / total_gt * 100:.2f}%"
)


print()
print("=" * 70)
print("JOINT ERROR RANKING")
print(
    "Higher error = worse"
)
print("=" * 70)

print(
    joint_df[
        [
            "rank",
            "joint",
            "mean_error",
            "medium_mean_error",
            "large_mean_error",
            "medium_minus_large",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 70)
print("MEDIUM VS LARGE")
print("=" * 70)

print(
    scale_summary.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(joint_csv)
print(scale_csv)
print(detail_csv)