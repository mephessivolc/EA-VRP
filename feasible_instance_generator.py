"""
instance_generator.py (geo‑bounded São Paulo)
============================================

Gera dados sintéticos (veículos, passageiros, depósito(s) e pontos de recarga)
num **bounding box geográfico**: por padrão a região metropolitana de São Paulo
(lat −24 a −23.4; lon −46.8 a −46.3).  Ainda garante que exista pelo menos
uma solução viável para o EA‑VRP e cria clusters de passageiros (1 a 5 pax).

Parâmetros ajustáveis
---------------------
* `bbox` – tupla `(lat_min, lat_max, lon_min, lon_max)`; default = São Paulo.
* `cluster_radius_km` – raio máximo (em km) entre passageiros do mesmo cluster.
* `dest_shift_km` – distância média do destino em relação à origem (km).

### Uso rápido
```python
bbox_sp = (-24.0, -23.4, -46.8, -46.3)
ig = InstanceGenerator(n_passengers=10, n_vehicles=3, seed=42,
                       bbox=bbox_sp, dest_shift_km=2.0)
vehicles, passengers, depots, rps = ig.build()
```

As coordenadas são valores **lat/lon** realistas, adequados para cálculos de
haversine, mapas, etc.
"""

from __future__ import annotations

import math
import random
import warnings
from typing import List, Tuple

import numpy as np

try:
    from utils import Passenger, Vehicle, Depot, RechargePoint
except ImportError as e:  # pragma: no cover
    raise ImportError("instance_generator.py requer utils.py no PYTHONPATH") from e

# ---------------- Global defaults -------------------------------------
MAX_CAPACITY = 5        # pax por veículo
CONSUMPTION_RATE = 0.10  # % bateria por km

DEFAULT_BBOX = (-24.0, -23.4, -46.8, -46.3)  # lat_min, lat_max, lon_min, lon_max (São Paulo)
CLUSTER_RADIUS_KM = 0.3   # ~300 m dentro do cluster
DEST_SHIFT_KM = 2.0       # ~2 km entre origem e destino

# Conversão aproximada grau↔km na latitude de SP (~−23.6°)
KM_PER_DEG_LAT = 111.32
KM_PER_DEG_LON = 111.32 * math.cos(math.radians(-23.6))  # ≈102.1 km


class InstanceGenerator:
    """Gera veículos + passageiros em bounding box geográfico, com clusters."""

    def __init__(
        self,
        n_passengers: int,
        n_vehicles: int,
        n_depots: int = 1,
        n_recharges: int = 1,
        seed: int | None = None,
        cluster_radius_km: float = CLUSTER_RADIUS_KM,
        dest_shift_km: float = DEST_SHIFT_KM,
        bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
    ) -> None:
        if n_depots < 1 or n_vehicles < 1 or n_passengers < 1:
            raise ValueError("n_depots, n_vehicles e n_passengers devem ser >=1")

        self.n_passengers = n_passengers
        self.n_vehicles = n_vehicles
        self.n_depots = n_depots
        self.n_recharges = n_recharges
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = bbox
        self.cluster_rad_lat = cluster_radius_km / KM_PER_DEG_LAT
        self.cluster_rad_lon = cluster_radius_km / KM_PER_DEG_LON
        self.dest_shift_lat = dest_shift_km / KM_PER_DEG_LAT
        self.dest_shift_lon = dest_shift_km / KM_PER_DEG_LON
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def build(self) -> Tuple[List[Vehicle], List[Passenger], List[Depot], List[RechargePoint]]:
        depots = self._make_depots()
        vehicles = self._make_vehicles()
        passengers = self._make_passengers(depots[0])
        rps = self._make_rps()
        return vehicles, passengers, depots, rps

    # ---------------- helpers -----------------------------------------
    def _make_depots(self) -> List[Depot]:
        # depos no centro do bbox para minimizar distâncias
        lat_c = (self.lat_min + self.lat_max) / 2
        lon_c = (self.lon_min + self.lon_max) / 2
        return [Depot(f"{i}", (lat_c, lon_c)) for i in range(self.n_depots)]

    def _make_vehicles(self) -> List[Vehicle]:
        lat_c = (self.lat_min + self.lat_max) / 2
        lon_c = (self.lon_min + self.lon_max) / 2
        return [Vehicle(f"{i+1}", (lat_c, lon_c)) for i in range(self.n_vehicles)]

    def _random_partition(self, total: int) -> List[int]:
        parts, rem = [], total
        while rem:
            size = self.rng.randint(1, min(MAX_CAPACITY, rem))
            parts.append(size)
            rem -= size
        self.rng.shuffle(parts)
        return parts

    def _make_passengers(self, depot: Depot) -> List[Passenger]:
        max_supported = self.n_vehicles * MAX_CAPACITY
        if self.n_passengers > max_supported:
            warnings.warn(f"Reduzindo passageiros para {max_supported} (capacidade)")
            n_pax = max_supported
        else:
            n_pax = self.n_passengers

        cluster_sizes = self._random_partition(n_pax)
        passengers: List[Passenger] = []
        pid = 1
        for size in cluster_sizes:
            base_lat = self.rng.uniform(self.lat_min, self.lat_max)
            base_lon = self.rng.uniform(self.lon_min, self.lon_max)

            # destino deslocado dest_shift_km para leste (aprox)
            dest_lat_base = base_lat + self.dest_shift_lat
            dest_lon_base = base_lon + self.dest_shift_lon

            # garante que destino ainda esteja no bbox (clip)
            dest_lat_base = min(max(dest_lat_base, self.lat_min), self.lat_max)
            dest_lon_base = min(max(dest_lon_base, self.lon_min), self.lon_max)

            for _ in range(size):
                lat_o = base_lat + self.np_rng.normal(0, self.cluster_rad_lat)
                lon_o = base_lon + self.np_rng.normal(0, self.cluster_rad_lon)
                lat_d = dest_lat_base + self.np_rng.normal(0, self.cluster_rad_lat)
                lon_d = dest_lon_base + self.np_rng.normal(0, self.cluster_rad_lon)

                # clip hard — evita escapar do bbox
                lat_o = min(max(lat_o, self.lat_min), self.lat_max)
                lon_o = min(max(lon_o, self.lon_min), self.lon_max)
                lat_d = min(max(lat_d, self.lat_min), self.lat_max)
                lon_d = min(max(lon_d, self.lon_min), self.lon_max)

                passengers.append(Passenger(f"P{pid}", (lat_o, lon_o), (lat_d, lon_d)))
                pid += 1
        return passengers

    def _make_rps(self) -> List[RechargePoint]:
        if self.n_recharges == 0:
            return []
        rps: List[RechargePoint] = []
        for i in range(self.n_recharges):
            # pontos igualmente espaçados ao longo da diagonal do bbox
            lat = self.lat_min + (self.lat_max - self.lat_min) * (i + 1) / (self.n_recharges + 1)
            lon = self.lon_min + (self.lon_max - self.lon_min) * (i + 1) / (self.n_recharges + 1)
            rps.append(RechargePoint(f"{i+1}", (lat, lon)))
        return rps


# ---------- autoteste --------------------------------------------------
if __name__ == "__main__":
    from utils import Group  # type: ignore
    from encoder import QUBOEncoder  # type: ignore
    from classical_solver import ClassicalVRPSolver  # type: ignore

    ig = InstanceGenerator(n_passengers=10, n_vehicles=3, seed=0)
    vehicles, passengers, depots, rps = ig.build()

    # agrupa por célula de 0.001° (~100 m)
    buckets = {}
    for p in passengers:
        key = (round(p.x_origin, 3), round(p.y_origin, 3))
        buckets.setdefault(key, []).append(p)
    groups = [Group(lst) for lst in buckets.values()]

    enc = QUBOEncoder(vehicles, groups, rps, depot=depots[0], penalty_lambda=5.0, penalty_mu=2.0)
    sol = ClassicalVRPSolver(enc)
    print(sol.best())
