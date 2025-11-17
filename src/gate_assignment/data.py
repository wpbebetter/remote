"""数据加载模块。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)

# data.py 位于 src/gate_assignment 下，parent[2] 定位到项目根目录
ROOT = Path(__file__).resolve().parents[2]
DATA_CLEAN = ROOT / "data" / "clean"

_TAXI_FILE = "PEK_stand_runway_directional_taxi_distances.csv"
_STANDS_FILE = "T3CDE_stands_clean.csv"
_CANDIDATE_FLIGHTS_FILE = "t3cde_candidate_flights_final_v2.backup.csv"


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


__all__ = [
    "load_taxi_distances",
    "load_stands",
    "load_candidate_flights",
]
