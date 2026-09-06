"""
使用 NumPy 生成实验用用户数据（含年龄、最大总时长 T），并保存为 CSV。
低龄段（40%）：年龄为浮点数且严格 < 15（未满 15 岁均视为低龄 / 14 岁段）。
高龄段（60%）：年龄 ∈ [15, 18]，浮点数保留一位小数。
各段内部年龄为右偏 Beta 映射；T（分钟）按年龄段分层正态：12–14 岁 N(45, 10)，15–18 岁 N(60, 15)（均值、标准差，单位：分钟）。
年龄分组 I：12–14 岁为 1，15–18 岁为 2（与年龄列一致：年龄 < 15 为 1，否则为 2）。
最大推送数量 K：K = ⌊T / t̄⌋，其中 t̄ 为视频平均时长（分钟），默认 2；T 为用户最大总时长（分钟）。
娱乐视频数量上限 A、教育视频数量下限 E：相对「最大推送数量」（列 K）占比向下取整，A = ⌊0.6×K⌋，E = ⌊0.2×K⌋。
最大承受风险 R：按年龄段分层正态——12–14 岁（本数据中年龄 < 15）R ~ N(20, 5)，15–18 岁 R ~ N(35, 8)，四舍五入保留一位小数后截断到 R ≥ 0（与 T 同序打乱）。
"""

import csv

import numpy as np

N_USERS = 1000
RNG_SEED = 42
OUTPUT_PATH = "users.csv"
# 右偏：α < β，密度在区间左端更高、右尾更长
BETA_A = 2.0
BETA_B = 5.0

# 最大总时长 T（分钟）：分层正态
T_MEAN_STD_YOUNG = (45.0, 10.0)  # 12–14 岁（本数据中年龄 < 15）
T_MEAN_STD_OLDER = (60.0, 15.0)  # 15–18 岁

# 最大推送数量 K：K = floor(T / AVG_VIDEO_DURATION_MIN)
AVG_VIDEO_DURATION_MIN = 2.0

# A、E：相对 K（最大推送数量）的比例，向下取整
ENT_VIDEO_CAP_FRAC = 0.6  # A = floor(K * 0.6)
EDU_VIDEO_FLOOR_FRAC = 0.2  # E = floor(K * 0.2)

# 最大承受风险 R：分层正态（均值、标准差与年龄段一致）
R_MEAN_STD_YOUNG = (20.0, 5.0)  # 12–14 岁
R_MEAN_STD_OLDER = (35.0, 8.0)  # 15–18 岁


def _right_skew_uniform(rng, low: float, high: float, size: int) -> np.ndarray:
    """Beta(α,β) 采样后线性映射到 [low, high]（闭区间）。"""
    u = rng.beta(BETA_A, BETA_B, size=size)
    return low + (high - low) * u


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    n_young = int(round(N_USERS * 0.4))
    n_older = N_USERS - n_young

    # 低龄：未满 15 岁（保留一位小数时上界为 14.9），在 [12.0, 14.9] 上右偏采样
    raw_young = _right_skew_uniform(rng, 12.0, 14.9, n_young)
    ages_young = np.round(raw_young, 1)
    ages_young = np.clip(ages_young, 12.0, 14.9)

    # 高龄：[15, 18]
    raw_older = _right_skew_uniform(rng, 15.0, 18.0, n_older)
    ages_older = np.round(raw_older, 1)
    ages_older = np.clip(ages_older, 15.0, 18.0)

    ages = np.concatenate([ages_young, ages_older])
    t_young = rng.normal(T_MEAN_STD_YOUNG[0], T_MEAN_STD_YOUNG[1], size=n_young)
    t_older = rng.normal(T_MEAN_STD_OLDER[0], T_MEAN_STD_OLDER[1], size=n_older)
    T = np.concatenate([t_young, t_older])
    T = np.maximum(np.round(T), 0).astype(np.int64)

    R_young = rng.normal(R_MEAN_STD_YOUNG[0], R_MEAN_STD_YOUNG[1], size=n_young)
    R_older = rng.normal(R_MEAN_STD_OLDER[0], R_MEAN_STD_OLDER[1], size=n_older)
    R = np.concatenate([R_young, R_older])

    perm = rng.permutation(N_USERS)
    ages = ages[perm]
    T = T[perm]
    R = R[perm]
    R = np.maximum(np.round(R, 1), 0.0)

    I = np.where(ages < 15, 1, 2).astype(np.int8)

    K = np.floor(T.astype(np.float64) / AVG_VIDEO_DURATION_MIN).astype(np.int64)
    A = np.floor(K.astype(np.float64) * ENT_VIDEO_CAP_FRAC).astype(np.int64)
    E = np.floor(K.astype(np.float64) * EDU_VIDEO_FLOOR_FRAC).astype(np.int64)

    user_ids = np.arange(1, N_USERS + 1, dtype=np.int64)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "age", "I", "T", "R", "K", "A", "E"])
        for uid, age, grp, t, r, k, a, e in zip(
            user_ids, ages, I, T, R, K, A, E, strict=True
        ):
            writer.writerow(
                [int(uid), f"{float(age):.1f}", int(grp), int(t), f"{float(r):.1f}", int(k), int(a), int(e)]
            )

    n_lt15 = np.sum(ages < 15)
    n_ge15 = np.sum(ages >= 15)
    print(f"已生成 {N_USERS} 条用户数据，保存至 {OUTPUT_PATH}")
    print(
        "分布校验（低龄 = 年龄 < 15）："
        f"{n_lt15} 人 ({100 * n_lt15 / N_USERS:.1f}%)，"
        f"高龄 {n_ge15} 人 ({100 * n_ge15 / N_USERS:.1f}%)"
    )
    print(f"低龄年龄范围 [{ages_young.min():.1f}, {ages_young.max():.1f}]，高龄 [{ages_older.min():.1f}, {ages_older.max():.1f}]")
    ty = T[ages < 15]
    to = T[ages >= 15]
    print(
        f"T（分钟）低龄 mean≈{ty.mean():.1f} std≈{ty.std(ddof=1):.1f}，"
        f"高龄 mean≈{to.mean():.1f} std≈{to.std(ddof=1):.1f}"
    )
    Ry = R[ages < 15]
    Ro = R[ages >= 15]
    print(
        f"R 低龄 N(20,5) mean≈{Ry.mean():.2f} std≈{Ry.std(ddof=1):.2f}；"
        f"高龄 N(35,8) mean≈{Ro.mean():.2f} std≈{Ro.std(ddof=1):.2f}"
    )


if __name__ == "__main__":
    main()
