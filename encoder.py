"""encoder.py
=================
QUBO encoder para EA‑VRP com **estações de recarga**.

* Variáveis y_{v,g}: veículo *v* atende grupo *g*.
* Variáveis z_{v,r}: veículo *v* visita estação *r* (no máximo uma por veículo).

Disponibiliza utilidades `to_matrix()` e `print_matrix()` para depuração.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from geografics import Distances
from grouping import EAVRPGroup
from utils import Vehicle

Idx = int
QUBO = Dict[Tuple[Idx, Idx], float]
Coord = Tuple[float, float]


class EA_VRP_QUBOEncoder:
    """Codifica EA‑VRP + recarga em formato QUBO."""

    # ------------------------------------------------------------------
    # Construtor
    # ------------------------------------------------------------------
    def __init__(self,
                 vehicles: List[Vehicle],
                 groups: List[EAVRPGroup],
                 stations: List[object] | None,
                 penalty_lambda: float | None = None,
                 penalty_mu: float | None = None,
                 depot: Coord | None = None,
                 metric: str = "euclidean") -> None:
        if not vehicles or not groups:
            raise ValueError("vehicles e groups não podem ser vazios")
        self.vehicles = vehicles
        self.groups = groups
        self.stations = stations or []
        self.metric_fn = getattr(Distances, metric)
        self.depot = depot or (0.0, 0.0)

        # custos
        self._cost_g = self._compute_group_costs()
        self._cost_r = self._compute_station_costs()
        all_costs = [c for row in self._cost_g + self._cost_r for c in row]
        max_c = max(all_costs) if all_costs else 1.0

        # penalidades
        self.lambda_ = penalty_lambda or 2 * max_c
        self.mu_ = penalty_mu or self.lambda_

        # variáveis
        self.index_map: Dict[Tuple[str, int, int], Idx] = {}
        idx = 0
        for v in range(len(self.vehicles)):
            for g in range(len(self.groups)):
                self.index_map[("G", v, g)] = idx; idx += 1
            for r in range(len(self.stations)):
                self.index_map[("R", v, r)] = idx; idx += 1
        self.reverse_map = {i: k for k, i in self.index_map.items()}

        self._Q: QUBO | None = None
        self._offset: float | None = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def num_qubits(self) -> int:
        """
        Número de qubits = número de variáveis ativas no QUBO atual (após shrink).
        """
        Q = self.Q  # garante construção do QUBO e aplicação do shrink
        indices = set()
        for i, j in Q.keys():
            indices.add(i)
            indices.add(j)

        return len(indices)

    @property
    def Q(self) -> QUBO:
        if self._Q is None:
            self._Q, self._offset = self._build_qubo()
        return self._Q

    @property
    def offset(self) -> float:
        if self._offset is None:
            _ = self.Q
        return float(self._offset)  # type: ignore

    def encode(self):
        return self.Q, self.offset, self.reverse_map

    # matrix for debug
    def to_matrix(self) -> np.ndarray:
        n = self.num_qubits
        mat = np.zeros((n, n))
        for (i, j), c in self.Q.items():
            mat[i, j] += c
            if i != j:
                mat[j, i] += c
        return mat

    def print_matrix(self, precision: int = 2):
        np.set_printoptions(precision=precision, suppress=True)
        print(self.to_matrix())
        np.set_printoptions()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_xy(self, obj):
        if isinstance(obj, (tuple, list)):
            return tuple(obj[:2])  # type: ignore[arg-type]
        for attr in ("coord", "coords", "location", "pos", "position"):
            xy = getattr(obj, attr, None)
            if xy is not None:
                return tuple(xy[:2])  # type: ignore[arg-type]
        raise TypeError("Station sem campo de coordenadas reconhecido")

    def _compute_group_costs(self):
        metric = self.metric_fn
        depot = self.depot[0]
        print(depot)
        costs: List[List[float]] = []
        for _ in self.vehicles:
            row = []
            for g in self.groups:
                go = g.origin()
                gd = g.destination()
                cost = metric(*depot, *go) + g.distance(metric=metric.__name__) + metric(*gd, *depot)
                row.append(cost)
            costs.append(row)
        return costs

    def _compute_station_costs(self):
        if not self.stations:
            return []
        metric = self.metric_fn; depot = self.depot[0]
        costs: List[List[float]] = []
        for _ in self.vehicles:
            row = []
            for st in self.stations:
                xy = self._extract_xy(st)
                row.append(metric(*depot, *xy) + metric(*xy, *depot))
            costs.append(row)
        return costs

    # ------------------------------------------------------------------
    # QUBO construction
    # ------------------------------------------------------------------
    def _build_qubo(self):
        Q: QUBO = {}; offset = 0.0

        # custos lineares
        for (kind, v, idx2), var in self.index_map.items():
            cost = self._cost_g[v][idx2] if kind == "G" else self._cost_r[v][idx2]
            Q[(var, var)] = Q.get((var, var), 0.0) + cost

        # restrição grupos
        lam = self.lambda_; n_v = len(self.vehicles); n_g = len(self.groups)
        for g in range(n_g):
            vars_g = [self.index_map[("G", v, g)] for v in range(n_v)]
            for var in vars_g:
                Q[(var, var)] = Q.get((var, var), 0.0) - lam
            for i in range(n_v):
                for j in range(i + 1, n_v):
                    vi, vj = vars_g[i], vars_g[j]
                    key = (vi, vj) if vi <= vj else (vj, vi)
                    Q[key] = Q.get(key, 0.0) + 2 * lam
            offset += lam

        # restrição recarga (<=1 por veículo)
        if self.stations:
            mu = self.mu_; n_r = len(self.stations)
            for v in range(n_v):
                vars_r = [self.index_map[("R", v, r)] for r in range(n_r)]
                for i in range(n_r):
                    for j in range(i + 1, n_r):
                        vi, vj = vars_r[i], vars_r[j]
                        key = (vi, vj) if vi <= vj else (vj, vi)
                        Q[key] = Q.get(key, 0.0) + 2 * mu

        return Q, offset


__all__ = ["EA_VRP_QUBOEncoder"]
