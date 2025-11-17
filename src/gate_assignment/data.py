"""数据加载模块。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# data.py 位于 src/gate_assignment 下，parent[2] 定位到项目根目录
ROOT = Path(__file__).resolve().parents[2]
DATA_CLEAN = ROOT / "data" / "clean"

_TAXI_FILE = "PEK_stand_runway_directional_taxi_distances.csv"
_STANDS_FILE = "T3CDE_stands_clean.csv"
_CANDIDATE_FLIGHTS_FILE = "t3cde_candidate_flights_final_v2.backup.csv"


@dataclass
class GateAssignmentInstance:
    """
    表示某一天的机位分配优化实例，包含数据表与已数值化的矩阵缓存。

    Attributes:
        date: 当天日期，去掉时间部分。
        flights: 当天候选航班子表。
        stands: 机位表子集（通常为全量机位）。
        taxi_distances: 机位-跑道滑行距离表。
        flight_ids: 每个航班的唯一标识。
        stand_ids: 可用机位 id 数组。
        arrival_true_min: 实际落地时间（单位：分钟，自当天 00:00 起算）。
        runway_codes: 每个航班的进港跑道字符串。
        taxi_cost_matrix: (n_flights, n_stands) 滑行距离矩阵。
        compat_matrix: (n_flights, n_stands) 可行性矩阵（0/1），当前暂设为 1。
    """

    date: pd.Timestamp
    flights: pd.DataFrame
    stands: pd.DataFrame
    taxi_distances: pd.DataFrame

    flight_ids: np.ndarray
    stand_ids: np.ndarray
    arrival_true_min: np.ndarray
    runway_codes: np.ndarray

    taxi_cost_matrix: np.ndarray
    compat_matrix: np.ndarray


_RUNWAY_COLUMN_MAP = {
    "18L": "to_18L_m",
    "36R": "to_36R_m",
    "01": "to_01_m",
    "1": "to_01_m",
    "19": "to_19_m",
    "18R": "to_18R_m",
    "36L": "to_36L_m",
}


def _runway_column_for(code: str) -> str:
    """根据跑道字符串返回对应的滑行距离列名。"""
    normalized = str(code).strip().upper().replace(" ", "")
    if normalized.isdigit() and len(normalized) == 1:
        normalized = normalized.zfill(2)
    column = _RUNWAY_COLUMN_MAP.get(normalized)
    if column is None:
        raise KeyError(f"未能识别跑道标识: {code!r}")
    return column


def _load_csv(filename: str, expected_columns: Iterable[str], encoding: str | None = None) -> pd.DataFrame:
    """
    通用的 CSV 读取与最小检查逻辑。

    Args:
        filename: 目标文件名，相对于 data/clean。
        expected_columns: 期望出现的关键列集合。
        encoding: 覆盖默认编码时使用，None 表示使用 pandas 默认值。

    Returns:
        pd.DataFrame: 通过基本校验的数据表。
    """
    path = DATA_CLEAN / filename
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    df = pd.read_csv(path, encoding=encoding)
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{filename} 缺少关键列: {missing}")
    if df.empty:
        raise ValueError(f"{filename} 为空，无法继续")

    LOGGER.info("Loaded %s with shape %s", filename, df.shape)
    return df


def load_taxi_distances() -> pd.DataFrame:
    """
    读取滑行距离矩阵。

    Returns:
        pd.DataFrame: 包含机位到各跑道方向滑行距离（米）的矩阵。
    """
    expected_columns = {
        "stand_id",
        "stand_ref",
        "to_18L_m",
        "to_36R_m",
        "to_01_m",
        "to_19_m",
        "to_18R_m",
        "to_36L_m",
    }
    return _load_csv(_TAXI_FILE, expected_columns)


def load_stands() -> pd.DataFrame:
    """
    读取机位元数据。

    Returns:
        pd.DataFrame: 包含机位编号、所属指廊、翼展约束等。
    """
    expected_columns = {
        "stand_id",
        "pier",
        "contact_remote",
        "wingspan_limit_m",
        "icao_code",
        "special_tag",
        "location_type",
    }
    return _load_csv(_STANDS_FILE, expected_columns)


def load_candidate_flights() -> pd.DataFrame:
    """
    读取候选航班集合。

    Returns:
        pd.DataFrame: 包含计划时间、航班号、机型与落地机位等字段。
    """
    expected_columns = {
        "日期",
        "航班号",
        "计划起飞时间",
        "计划落地时间",
        "机型",
        "起飞机场",
        "航空公司",
        "进港跑道",
        "landing_bay",
        "landing_bay_str",
    }
    return _load_csv(_CANDIDATE_FLIGHTS_FILE, expected_columns, encoding="utf-8-sig")


def build_daily_instances(min_flights_per_day: int = 1) -> List[GateAssignmentInstance]:
    """
    从 clean CSV 构造“按天”的机位分配实例列表。

    Args:
        min_flights_per_day: 若某日航班数少于该值则跳过，避免噪声样本。

    Returns:
        List[GateAssignmentInstance]: 依日期排序的实例列表。
    """

    if min_flights_per_day < 1:
        raise ValueError("min_flights_per_day 至少为 1")

    flights_df = load_candidate_flights().copy()
    stands_df = load_stands().copy()
    taxi_df = load_taxi_distances().copy()

    flights_df["日期"] = pd.to_datetime(flights_df["日期"], errors="coerce")
    flights_df["实际落地时间"] = pd.to_datetime(
        flights_df["实际落地时间"], errors="coerce"
    )
    flights_df["runway_str"] = (
        flights_df["runway_str"].astype(str).str.strip().str.upper()
    )
    invalid_runway = flights_df["runway_str"].isin({"", "NAN"})
    flights_df.loc[invalid_runway, "runway_str"] = np.nan
    flights_df = flights_df.dropna(subset=["日期", "实际落地时间", "runway_str"])
    flights_df["date_only"] = flights_df["日期"].dt.normalize()

    taxi_df["stand_ref"] = taxi_df["stand_ref"].astype(str)
    taxi_lookup = taxi_df.set_index("stand_ref")
    stand_ids = stands_df["stand_id"].to_numpy()
    stand_refs = stands_df["stand_id"].astype(str)
    try:
        taxi_for_stands = taxi_lookup.loc[stand_refs]
    except KeyError as exc:  # pragma: no cover
        missing = set(stand_refs) - set(taxi_lookup.index)
        raise KeyError(f"下列机位缺少滑行距离: {missing}") from exc
    taxi_column_cache = {
        col: taxi_for_stands[col].to_numpy(dtype=float)
        for col in _RUNWAY_COLUMN_MAP.values()
        if col in taxi_for_stands.columns
    }
    missing_columns = set(_RUNWAY_COLUMN_MAP.values()) - set(taxi_column_cache)
    if missing_columns:  # pragma: no cover
        raise KeyError(f"滑行距离文件缺少列: {missing_columns}")

    instances: List[GateAssignmentInstance] = []
    for date_key, df_day in flights_df.groupby("date_only"):
        if len(df_day) < min_flights_per_day:
            continue
        df_day = df_day.sort_values("实际落地时间").reset_index(drop=True)
        arrivals = pd.to_datetime(df_day["实际落地时间"], errors="coerce")
        runway_codes = df_day["runway_str"].to_numpy()
        day_start = pd.Timestamp(date_key).normalize()
        arrival_minutes = ((arrivals - day_start).dt.total_seconds() / 60.0).to_numpy()

        flight_ids = (
            df_day["date_only"].dt.strftime("%Y%m%d")
            + "_"
            + df_day["航班号"].astype(str).str.strip()
        ).to_numpy()

        n_flights = len(df_day)
        n_stands = len(stand_ids)
        taxi_cost_matrix = np.zeros((n_flights, n_stands), dtype=float)
        for i, runway_code in enumerate(runway_codes):
            column = _runway_column_for(runway_code)
            taxi_cost_matrix[i, :] = taxi_column_cache[column]
        compat_matrix = np.ones((n_flights, n_stands), dtype=np.int8)
        # TODO: 根据机型、翼展、区域规则生成真实的兼容矩阵

        flights_day = (
            df_day.drop(columns=["date_only"])
            .reset_index(drop=True)
            .assign(arrival_true_min=arrival_minutes, runway_code=runway_codes)
        )
        instance = GateAssignmentInstance(
            date=pd.Timestamp(date_key).normalize(),
            flights=flights_day,
            stands=stands_df,
            taxi_distances=taxi_df,
            flight_ids=flight_ids,
            stand_ids=stand_ids,
            arrival_true_min=arrival_minutes,
            runway_codes=runway_codes,
            taxi_cost_matrix=taxi_cost_matrix,
            compat_matrix=compat_matrix,
        )
        instances.append(instance)

    instances.sort(key=lambda inst: inst.date)
    LOGGER.info("构造出 %d 个日实例（min_flights_per_day=%d）", len(instances), min_flights_per_day)
    return instances


__all__ = [
    "load_taxi_distances",
    "load_stands",
    "load_candidate_flights",
    "GateAssignmentInstance",
    "build_daily_instances",
]
