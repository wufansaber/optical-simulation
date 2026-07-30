"""透镜处方 CSV 的读取与校验。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Dict, Optional, Tuple

from .interactive_model import InteractiveLens


_EMPTY_FOCAL_VALUES = {"", "--", "none", "inf", "+inf", "infinity"}
_OUTPUT_MARKERS = {"target", "output", "workpiece", "工件面", "输出面"}
_COLUMN_ALIASES = {
    "name": ("name", "lens_name", "名称"),
    "z_mm": ("z_mm", "z/mm", "z", "位置/mm"),
    "focal_x_mm": ("focal_x_mm", "fx_mm", "fx/mm", "fx"),
    "focal_y_mm": ("focal_y_mm", "fy_mm", "fy/mm", "fy"),
}


@dataclass(frozen=True)
class LensCsvImport:
    lenses: Tuple[InteractiveLens, ...]
    observation_z_m: Optional[float] = None


def _normalized_header(text: Optional[str]) -> str:
    return "" if text is None else text.strip().lower()


def _resolve_columns(fieldnames) -> Dict[str, str]:
    normalized = {_normalized_header(name): name for name in (fieldnames or [])}
    resolved: Dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                resolved[canonical] = normalized[alias]
                break
        if canonical not in resolved:
            raise ValueError(
                "CSV 缺少必要列 {}。支持的标准表头为："
                "name,z_mm,focal_x_mm,focal_y_mm。".format(canonical)
            )
    return resolved


def _optional_focal_m(text: Optional[str]) -> Optional[float]:
    value = "" if text is None else text.strip().lower()
    if value in _EMPTY_FOCAL_VALUES:
        return None
    return float(value) * 1e-3


def load_lenses_csv(path: Path) -> LensCsvImport:
    """读取软件透镜 CSV；校验成功后返回完整透镜列表。"""

    csv_path = Path(path)
    lenses = []
    observation_z_m: Optional[float] = None
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = _resolve_columns(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            try:
                name = (row.get(columns["name"]) or "").strip()
                z_m = float((row.get(columns["z_mm"]) or "").strip()) * 1e-3
                focal_x_m = _optional_focal_m(row.get(columns["focal_x_mm"]))
                focal_y_m = _optional_focal_m(row.get(columns["focal_y_mm"]))
                if name.lower() in _OUTPUT_MARKERS and focal_x_m is None and focal_y_m is None:
                    if not isfinite(z_m) or z_m < 0.0:
                        raise ValueError("TARGET 输出位置必须是非负有限数值")
                    observation_z_m = z_m
                    continue
                if focal_x_m is None and focal_y_m is None:
                    raise ValueError("至少需要一个非空的 X 或 Y 焦距")
                lenses.append(
                    InteractiveLens(
                        name=name,
                        z_m=z_m,
                        focal_x_m=focal_x_m,
                        focal_y_m=focal_y_m,
                    )
                )
            except Exception as error:
                raise ValueError("CSV 第 {} 行无效：{}。".format(row_number, error)) from error
    if not lenses:
        raise ValueError("CSV 中没有可导入的透镜记录。")
    return LensCsvImport(tuple(lenses), observation_z_m)
