"""基于 ABCD 矩阵的近轴光学元件。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import List, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError("{} 必须是有限数值。".format(name))


@dataclass(frozen=True)
class Ray:
    """近轴光线，由光轴高度和相对光轴的角度表示。"""

    height: float
    angle: float

    def __post_init__(self) -> None:
        _require_finite("光线高度", self.height)
        _require_finite("光线角度", self.angle)

    def as_vector(self) -> FloatArray:
        return np.array([self.height, self.angle], dtype=np.float64)

    @classmethod
    def from_vector(cls, vector: FloatArray) -> Ray:
        return cls(height=float(vector[0]), angle=float(vector[1]))


class OpticalElement(Protocol):
    @property
    def matrix(self) -> FloatArray:
        ...


@dataclass(frozen=True)
class FreeSpace:
    """长度为 distance 的均匀自由空间。"""

    distance: float

    def __post_init__(self) -> None:
        _require_finite("传播距离", self.distance)
        if self.distance < 0:
            raise ValueError("传播距离不能为负数。")

    @property
    def matrix(self) -> FloatArray:
        return np.array([[1.0, self.distance], [0.0, 1.0]], dtype=np.float64)


@dataclass(frozen=True)
class ThinLens:
    """焦距为 focal_length 的理想薄透镜。"""

    focal_length: float

    def __post_init__(self) -> None:
        _require_finite("焦距", self.focal_length)
        if self.focal_length == 0:
            raise ValueError("薄透镜焦距不能为零。")

    @property
    def matrix(self) -> FloatArray:
        return np.array(
            [[1.0, 0.0], [-1.0 / self.focal_length, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class TracePoint:
    z: float
    ray: Ray
    element: str


@dataclass
class OpticalSystem:
    """按实际传播顺序组成的近轴光学系统。"""

    elements: List[OpticalElement] = field(default_factory=list)

    @property
    def matrix(self) -> FloatArray:
        result = np.eye(2, dtype=np.float64)
        for element in self.elements:
            result = element.matrix @ result
        return result

    def propagate(self, ray: Ray) -> Ray:
        return Ray.from_vector(self.matrix @ ray.as_vector())

    def trace(self, ray: Ray) -> List[TracePoint]:
        current = ray
        z = 0.0
        points = [TracePoint(z=z, ray=current, element="输入面")]
        for element in self.elements:
            current = Ray.from_vector(element.matrix @ current.as_vector())
            if isinstance(element, FreeSpace):
                z += element.distance
                label = "自由空间 {:g} m".format(element.distance)
            elif isinstance(element, ThinLens):
                label = "薄透镜 f={:g} m".format(element.focal_length)
            else:
                label = type(element).__name__
            points.append(TracePoint(z=z, ray=current, element=label))
        return points

    def lens_positions(self) -> List[Tuple[float, ThinLens]]:
        z = 0.0
        positions = []
        for element in self.elements:
            if isinstance(element, FreeSpace):
                z += element.distance
            elif isinstance(element, ThinLens):
                positions.append((z, element))
        return positions

