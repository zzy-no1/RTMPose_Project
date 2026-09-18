import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 路径
# ============================================================

DETAIL_CSV = r"analysis\error_analysis\matched_joint_details.csv"

COCO_JSON = (
    r"data\coco\annotations"
    r"\person_keypoints_val2017.json"
)

OUT_DIR = r"analysis\error_distribution"
FIG_DIR = os.path.join(OUT_DIR, "figures")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# COCO 17 关键点顺序
# ============================================================

COCO_KEYPOINTS = [
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
# 读取误差明细
# ============================================================

print("Loading matched joint details...")

df = pd.read_csv(DETAIL_CSV)

print("Rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# 读取 COCO annotation，补 visibility
#
# COCO:
# v = 0 -> not labeled
# v = 1 -> labeled but not visible
# v = 2 -> visible
# ============================================================

print("Loading COCO annotations...")

with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

ann_dict = {
    int(ann["id"]): ann
    for ann in coco["annotations"]
}


def get_visibility(ann_id, joint_name):
    ann = ann_dict.get(int(ann_id))

    if ann is None:
        return np.nan

    if joint_name not in COCO_KEYPOINTS:
        return np.nan

    joint_idx = COCO_KEYPOINTS.index(joint_name)

    keypoints = ann["keypoints"]

    visibility = keypoints[
        joint_idx * 3 + 2
    ]

    return int(visibility)


print("Adding visibility information...")

df["visibility"] = [
    get_visibility(ann_id, joint)
    for ann_id, joint in zip(
        df["ann_id"],
        df["joint"]
    )
]


def visibility_name(v):
    if v == 2:
        return "visible"
    elif v == 1:
        return "occluded"
    elif v == 0:
        return "not_labeled"
    else:
        return "unknown"


df["visibility_group"] = df[
    "visibility"
].apply(visibility_name)


# ============================================================
# 基础检查
# ============================================================

print()
print("=" * 70)
print("VISIBILITY COUNTS")
print("=" * 70)

print(
    df["visibility_group"]
    .value_counts(dropna=False)
)


# ============================================================
# 分位数统计函数
# ============================================================

def summarize_errors(group):

    x = group[
        "normalized_error"
    ].dropna()

    return pd.Series({
        "count": len(x),
        "mean": x.mean(),
        "median": x.median(),
        "p75": x.quantile(0.75),
        "p90": x.quantile(0.90),
        "p95": x.quantile(0.95),
        "p99": x.quantile(0.99),
        "max": x.max(),
        "mean_pixel_error":
            group["pixel_error"].mean(),
        "mean_bbox_iou":
            group["bbox_iou"].mean(),
        "mean_bbox_score":
            group["bbox_score"].mean(),
    })


# ============================================================
# 1. 每个 Joint 的误差分布
# ============================================================

joint_summary = (
    df.groupby("joint")
    .apply(summarize_errors)
    .reset_index()
)

joint_summary = joint_summary.sort_values(
    "mean",
    ascending=False
)

joint_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "joint_error_distribution.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 2. Medium vs Large
# ============================================================

scale_summary = (
    df.groupby("scale_group")
    .apply(summarize_errors)
    .reset_index()
)

scale_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "scale_error_distribution.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 3. Visible vs Occluded
# ============================================================

visibility_df = df[
    df["visibility_group"].isin(
        ["visible", "occluded"]
    )
].copy()

visibility_summary = (
    visibility_df
    .groupby("visibility_group")
    .apply(summarize_errors)
    .reset_index()
)

visibility_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "visibility_error_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 4. Joint × Scale
# ============================================================

joint_scale_summary = (
    df.groupby(
        ["joint", "scale_group"]
    )
    .apply(summarize_errors)
    .reset_index()
)

joint_scale_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "joint_scale_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. Joint × Visibility
# ============================================================

joint_visibility_summary = (
    visibility_df
    .groupby(
        ["joint", "visibility_group"]
    )
    .apply(summarize_errors)
    .reset_index()
)

joint_visibility_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "joint_visibility_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 6. Catastrophic Error 分析
#
# 不人为指定阈值。
# 先使用整体误差分布 P95 / P99。
# ============================================================

all_errors = df[
    "normalized_error"
].dropna()

p75 = all_errors.quantile(0.75)
p90 = all_errors.quantile(0.90)
p95 = all_errors.quantile(0.95)
p99 = all_errors.quantile(0.99)

print()
print("=" * 70)
print("GLOBAL ERROR QUANTILES")
print("=" * 70)

print(f"P75: {p75:.6f}")
print(f"P90: {p90:.6f}")
print(f"P95: {p95:.6f}")
print(f"P99: {p99:.6f}")
print(f"MAX: {all_errors.max():.6f}")


def error_level(x):
    if x < p75:
        return "normal"
    elif x < p90:
        return "moderate"
    elif x < p95:
        return "hard"
    elif x < p99:
        return "very_hard"
    else:
        return "catastrophic"


df["error_level"] = df[
    "normalized_error"
].apply(error_level)


error_level_summary = (
    df["error_level"]
    .value_counts()
    .rename_axis("error_level")
    .reset_index(name="count")
)

error_level_summary[
    "ratio"
] = (
    error_level_summary["count"]
    / len(df)
)

error_level_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "error_level_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. 高误差是否与 BBox IoU 有关
# ============================================================

bbox_error_summary = (
    df.groupby("error_level")
    .agg(
        count=("normalized_error", "size"),
        mean_error=(
            "normalized_error",
            "mean"
        ),
        median_error=(
            "normalized_error",
            "median"
        ),
        mean_bbox_iou=(
            "bbox_iou",
            "mean"
        ),
        median_bbox_iou=(
            "bbox_iou",
            "median"
        ),
        mean_bbox_score=(
            "bbox_score",
            "mean"
        ),
    )
    .reset_index()
)

bbox_error_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "bbox_error_relationship.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. 图1：全局误差分布 Histogram
# ============================================================

plt.figure(figsize=(8, 5))

plot_errors = all_errors[
    all_errors <= p99
]

plt.hist(
    plot_errors,
    bins=80
)

plt.axvline(
    p75,
    linestyle="--",
    label=f"P75={p75:.3f}"
)

plt.axvline(
    p90,
    linestyle="--",
    label=f"P90={p90:.3f}"
)

plt.axvline(
    p95,
    linestyle="--",
    label=f"P95={p95:.3f}"
)

plt.xlabel(
    "Normalized Keypoint Error"
)
plt.ylabel("Count")
plt.title(
    "RTMPose-M Keypoint Error Distribution"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "01_error_distribution.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 9. 图2：各关节 Median / P90 / P95
# ============================================================

plot_df = joint_summary.sort_values(
    "median",
    ascending=False
)

x = np.arange(len(plot_df))
width = 0.25

plt.figure(figsize=(14, 6))

plt.bar(
    x - width,
    plot_df["median"],
    width,
    label="Median"
)

plt.bar(
    x,
    plot_df["p90"],
    width,
    label="P90"
)

plt.bar(
    x + width,
    plot_df["p95"],
    width,
    label="P95"
)

plt.xticks(
    x,
    plot_df["joint"],
    rotation=60,
    ha="right"
)

plt.ylabel(
    "Normalized Keypoint Error"
)

plt.title(
    "Joint Error Distribution: "
    "Median / P90 / P95"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "02_joint_median_p90_p95.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 10. 图3：Medium vs Large
# ============================================================

scale_pivot = (
    joint_scale_summary
    .pivot(
        index="joint",
        columns="scale_group",
        values="mean"
    )
)

scale_pivot = scale_pivot.loc[
    joint_summary["joint"]
]

x = np.arange(
    len(scale_pivot)
)

width = 0.36

plt.figure(figsize=(14, 6))

if "medium" in scale_pivot.columns:

    plt.bar(
        x - width / 2,
        scale_pivot["medium"],
        width,
        label="Medium"
    )

if "large" in scale_pivot.columns:

    plt.bar(
        x + width / 2,
        scale_pivot["large"],
        width,
        label="Large"
    )

plt.xticks(
    x,
    scale_pivot.index,
    rotation=60,
    ha="right"
)

plt.ylabel(
    "Mean Normalized Error"
)

plt.title(
    "Joint Error: Medium vs Large"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "03_medium_vs_large.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 11. 图4：Visible vs Occluded
# ============================================================

vis_pivot = (
    joint_visibility_summary
    .pivot(
        index="joint",
        columns="visibility_group",
        values="mean"
    )
)

available_order = [
    j
    for j in joint_summary["joint"]
    if j in vis_pivot.index
]

vis_pivot = vis_pivot.loc[
    available_order
]

x = np.arange(
    len(vis_pivot)
)

width = 0.36

plt.figure(figsize=(14, 6))

if "visible" in vis_pivot.columns:

    plt.bar(
        x - width / 2,
        vis_pivot["visible"],
        width,
        label="Visible"
    )

if "occluded" in vis_pivot.columns:

    plt.bar(
        x + width / 2,
        vis_pivot["occluded"],
        width,
        label="Occluded"
    )

plt.xticks(
    x,
    vis_pivot.index,
    rotation=60,
    ha="right"
)

plt.ylabel(
    "Mean Normalized Error"
)

plt.title(
    "Joint Error: Visible vs Occluded"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "04_visible_vs_occluded.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 12. 图5：BBox IoU vs Error
# ============================================================

sample_df = df.copy()

# 防止点太多，随机抽样
if len(sample_df) > 12000:
    sample_df = sample_df.sample(
        n=12000,
        random_state=42
    )

plt.figure(figsize=(8, 5))

plt.scatter(
    sample_df["bbox_iou"],
    sample_df["normalized_error"],
    s=8,
    alpha=0.25
)

plt.ylim(
    0,
    min(
        p99 * 1.2,
        sample_df[
            "normalized_error"
        ].max()
    )
)

plt.xlabel("BBox IoU")
plt.ylabel(
    "Normalized Keypoint Error"
)

plt.title(
    "BBox IoU vs Keypoint Error"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "05_bbox_iou_vs_error.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 输出控制台摘要
# ============================================================

print()
print("=" * 70)
print("JOINT ERROR DISTRIBUTION")
print("=" * 70)

print(
    joint_summary[
        [
            "joint",
            "mean",
            "median",
            "p90",
            "p95",
            "p99",
            "max",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 70)
print("SCALE ERROR DISTRIBUTION")
print("=" * 70)

print(
    scale_summary.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("VISIBILITY ERROR DISTRIBUTION")
print("=" * 70)

print(
    visibility_summary.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("ERROR LEVEL SUMMARY")
print("=" * 70)

print(
    error_level_summary.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("ERROR LEVEL VS BBOX IOU")
print("=" * 70)

print(
    bbox_error_summary.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

for filename in [
    "joint_error_distribution.csv",
    "scale_error_distribution.csv",
    "visibility_error_summary.csv",
    "joint_scale_summary.csv",
    "joint_visibility_summary.csv",
    "error_level_summary.csv",
    "bbox_error_relationship.csv",
]:

    print(
        os.path.join(
            OUT_DIR,
            filename
        )
    )

print()

print(
    "Figures saved to:",
    FIG_DIR
)