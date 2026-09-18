import os
import json

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 路径
# ============================================================

DETAIL_CSV = r"analysis\error_analysis\matched_joint_details.csv"

COCO_JSON = (
    r"data\coco\annotations"
    r"\person_keypoints_val2017.json"
)

IMAGE_DIR = r"data\coco\val2017"

OUT_DIR = r"analysis\bad_cases"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 参数
# ============================================================

TOP_K = 50

# 如果只想分析某些关节，可修改这里。
# None 表示所有关节一起排序。
TARGET_JOINTS = None

# 例如只看四肢难点：
# TARGET_JOINTS = [
#     "left_wrist", "right_wrist",
#     "left_ankle", "right_ankle",
#     "left_knee", "right_knee",
#     "left_hip", "right_hip"
# ]


# ============================================================
# 读取误差结果
# ============================================================

print("Loading error analysis CSV...")

df = pd.read_csv(DETAIL_CSV)

print("Total joint samples:", len(df))


if TARGET_JOINTS is not None:
    df = df[
        df["joint"].isin(TARGET_JOINTS)
    ].copy()


# 按 normalized_error 从大到小排列
df = df.sort_values(
    "normalized_error",
    ascending=False
)

top_df = df.head(TOP_K).copy()


# ============================================================
# 读取 COCO annotation
# ============================================================

print("Loading COCO annotations...")

with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)


ann_dict = {
    int(ann["id"]): ann
    for ann in coco["annotations"]
}


# ============================================================
# 可视化函数
# ============================================================

def draw_point(
    image,
    x,
    y,
    radius,
    color,
    thickness=-1
):
    cv2.circle(
        image,
        (int(round(x)), int(round(y))),
        radius,
        color,
        thickness,
        lineType=cv2.LINE_AA
    )


def draw_cross(
    image,
    x,
    y,
    color,
    size=6,
    thickness=2
):
    x = int(round(x))
    y = int(round(y))

    cv2.line(
        image,
        (x - size, y - size),
        (x + size, y + size),
        color,
        thickness,
        cv2.LINE_AA
    )

    cv2.line(
        image,
        (x - size, y + size),
        (x + size, y - size),
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 开始生成 bad cases
# ============================================================

saved_rows = []

for rank, (_, row) in enumerate(
    top_df.iterrows(),
    start=1
):

    image_id = int(row["image_id"])
    ann_id = int(row["ann_id"])

    joint = row["joint"]
    scale_group = row["scale_group"]

    gt_x = float(row["gt_x"])
    gt_y = float(row["gt_y"])

    pred_x = float(row["pred_x"])
    pred_y = float(row["pred_y"])

    pixel_error = float(row["pixel_error"])
    normalized_error = float(
        row["normalized_error"]
    )

    bbox_iou = float(row["bbox_iou"])

    image_name = f"{image_id:012d}.jpg"

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    image = cv2.imread(image_path)

    if image is None:
        print(
            "Failed to read:",
            image_path
        )
        continue


    # ========================================================
    # 获取 GT person bbox
    # ========================================================

    ann = ann_dict.get(ann_id)

    if ann is not None:

        x, y, w, h = ann["bbox"]

        cv2.rectangle(
            image,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            (255, 255, 0),
            2
        )


    # ========================================================
    # GT 与预测关键点
    #
    # GT：绿色圆点
    # Pred：红色叉
    # GT → Pred：黄色连线
    # ========================================================

    draw_point(
        image,
        gt_x,
        gt_y,
        radius=6,
        color=(0, 255, 0)
    )

    draw_cross(
        image,
        pred_x,
        pred_y,
        color=(0, 0, 255),
        size=7,
        thickness=2
    )

    cv2.line(
        image,
        (
            int(round(gt_x)),
            int(round(gt_y))
        ),
        (
            int(round(pred_x)),
            int(round(pred_y))
        ),
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # 左上角信息
    # ========================================================

    text_lines = [
        f"Rank: {rank}",
        f"Joint: {joint}",
        f"Scale: {scale_group}",
        f"Norm error: {normalized_error:.4f}",
        f"Pixel error: {pixel_error:.1f}px",
        f"BBox IoU: {bbox_iou:.3f}",
    ]

    y0 = 30

    for text in text_lines:

        cv2.putText(
            image,
            text,
            (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            text,
            (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        y0 += 28


    # 图例
    legend_y = y0 + 10

    cv2.putText(
        image,
        "Green = GT",
        (20, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        "Red X = Prediction",
        (20, legend_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # 保存
    # ========================================================

    safe_joint = joint.replace("/", "_")

    output_name = (
        f"{rank:03d}_"
        f"{safe_joint}_"
        f"{scale_group}_"
        f"img_{image_id}.jpg"
    )

    output_path = os.path.join(
        OUT_DIR,
        output_name
    )

    cv2.imwrite(
        output_path,
        image
    )

    saved_rows.append(
        {
            "rank": rank,
            "image_id": image_id,
            "ann_id": ann_id,
            "joint": joint,
            "scale_group": scale_group,
            "normalized_error":
                normalized_error,
            "pixel_error":
                pixel_error,
            "bbox_iou":
                bbox_iou,
            "output_file":
                output_name,
        }
    )

    print(
        f"[{rank}/{TOP_K}]",
        output_name
    )


# ============================================================
# 保存 Top K 清单
# ============================================================

summary_path = os.path.join(
    OUT_DIR,
    "top_bad_cases.csv"
)

pd.DataFrame(
    saved_rows
).to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Bad case images saved to: {OUT_DIR}"
)

print(
    f"Summary saved to: {summary_path}"
)