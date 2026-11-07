"""Top-10 urban agglomeration registry for multi-city placement transfer runs.

These are core urban bounding boxes (not full metro sprawl) sized so OSM station
queries and Kolesar placement stay tractable. Demand is a population proxy —
not real fire-service incidents. See ``docs/megacities.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Megacity:
    """One megacity core used for OSM + proxy-demand placement."""

    id: str
    name: str
    # west, south, east, north (degrees)
    bbox: tuple[float, float, float, float]
    # lon, lat
    center: tuple[float, float]
    # approx 2024 agglomeration rank / population hint for UI only
    pop_millions: float
    notes: str = ""

    @property
    def raw_dir(self) -> Path:
        return ROOT / "data" / "raw" / "megacities" / self.id

    @property
    def fig_dir(self) -> Path:
        return ROOT / "data" / "figures" / "megacities" / self.id

    @property
    def fire_stations_path(self) -> Path:
        return self.raw_dir / "fire_stations.csv"

    @property
    def incidents_path(self) -> Path:
        return self.raw_dir / "incidents_proxy.csv"

    @property
    def graph_path(self) -> Path:
        return self.raw_dir / "drive.graphml"


# Core bboxes ~50–80 km across, clipped to dense urban cores.
MEGACITIES: dict[str, Megacity] = {
    "tokyo": Megacity(
        id="tokyo",
        name="Tokyo",
        bbox=(139.40, 35.45, 139.95, 35.85),
        center=(139.69, 35.68),
        pop_millions=37.0,
        notes="Tokyo–Yokohama core",
    ),
    "delhi": Megacity(
        id="delhi",
        name="Delhi",
        bbox=(76.85, 28.40, 77.45, 28.90),
        center=(77.21, 28.61),
        pop_millions=32.0,
        notes="National Capital Region core",
    ),
    "shanghai": Megacity(
        id="shanghai",
        name="Shanghai",
        bbox=(121.20, 31.05, 121.75, 31.45),
        center=(121.47, 31.23),
        pop_millions=29.0,
    ),
    "dhaka": Megacity(
        id="dhaka",
        name="Dhaka",
        bbox=(90.25, 23.65, 90.55, 23.95),
        center=(90.39, 23.81),
        pop_millions=23.0,
    ),
    "cairo": Megacity(
        id="cairo",
        name="Cairo",
        bbox=(31.10, 29.90, 31.50, 30.25),
        center=(31.24, 30.04),
        pop_millions=22.0,
    ),
    "sao_paulo": Megacity(
        id="sao_paulo",
        name="São Paulo",
        bbox=(-46.85, -23.75, -46.40, -23.40),
        center=(-46.63, -23.55),
        pop_millions=22.0,
    ),
    "mexico_city": Megacity(
        id="mexico_city",
        name="Mexico City",
        bbox=(-99.30, 19.25, -98.95, 19.55),
        center=(-99.13, 19.43),
        pop_millions=22.0,
    ),
    "beijing": Megacity(
        id="beijing",
        name="Beijing",
        bbox=(116.15, 39.75, 116.65, 40.10),
        center=(116.40, 39.90),
        pop_millions=21.0,
    ),
    "mumbai": Megacity(
        id="mumbai",
        name="Mumbai",
        bbox=(72.75, 18.90, 73.05, 19.30),
        center=(72.88, 19.08),
        pop_millions=21.0,
    ),
    "osaka": Megacity(
        id="osaka",
        name="Osaka",
        bbox=(135.35, 34.55, 135.70, 34.85),
        center=(135.50, 34.69),
        pop_millions=19.0,
        notes="Osaka–Kobe–Kyoto core slice around Osaka",
    ),
}


def list_megacities() -> list[str]:
    return list(MEGACITIES.keys())


def get_megacity(city_id: str) -> Megacity:
    key = str(city_id).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in MEGACITIES:
        raise KeyError(f"Unknown megacity {city_id!r}. Known: {list_megacities()}")
    return MEGACITIES[key]


def iter_megacities(ids: Iterable[str] | None = None) -> list[Megacity]:
    if ids is None:
        return [MEGACITIES[k] for k in list_megacities()]
    return [get_megacity(i) for i in ids]
