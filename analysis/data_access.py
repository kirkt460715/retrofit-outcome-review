"""读取数据目录、解析日期与周区间。

所有日期都是北京时间，周区间为左闭右开 [周一, 下周一)。
对外一律使用 ISO 日期字符串，避免不同机器上的时区解释差异。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Mapping

from inspect_records import read_collection

ONE_WEEK = timedelta(days=7)


def parse_date(value: str) -> date:
    """把空串保留为 None 的语义由调用方处理；这里只解析非空日期。"""
    if not value:
        raise ValueError("空日期不能解析")
    return date.fromisoformat(value)


def week_end(week: str) -> date:
    """周起点加七天，作为左闭右开区间的右端。"""
    return parse_date(week) + ONE_WEEK


@dataclass(frozen=True)
class Tables:
    """一次载入的全部集合，保持只读以免分析中途被改写。"""

    buildings: tuple[Mapping[str, str], ...]
    households: tuple[Mapping[str, str], ...]
    phases: tuple[Mapping[str, str], ...]
    connections: tuple[Mapping[str, str], ...]
    observations: tuple[Mapping[str, str], ...]
    other_projects: tuple[Mapping[str, str], ...]
    proposals: tuple[Mapping[str, str], ...]

    @classmethod
    def load(cls, directory: Path) -> "Tables":
        names = (
            "buildings",
            "households",
            "phases",
            "connections",
            "observations",
            "other_projects",
            "proposals",
        )
        loaded = {name: tuple(read_collection(name, directory)[0]) for name in names}
        return cls(**loaded)

    def weeks(self) -> list[str]:
        return sorted({row["week"] for row in self.observations})


def index(rows: Iterable[Mapping[str, str]], key: str) -> dict[str, Mapping[str, str]]:
    """按主键建立索引，重复键在装载阶段已拒绝。"""
    return {row[key]: row for row in rows}
