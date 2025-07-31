"""grouping.py
================
Módulo responsável por formar grupos geográficos de 1‑5 passageiros a partir
de uma lista de instâncias ``Passenger`` e retornar objetos ``EAVRPGroup`` com
métricas úteis para o restante do pipeline EA‑VRP.

Observação importante
--------------------
As funções em ``geografics.Distances`` (``euclidean``, ``haversine``, etc.)
recebem **quatro valores escalares** – ``x1, y1, x2, y2``. Portanto, sempre que
invocamos um método da classe ``Distances`` devemos *desempacotar* as
coordenadas (lat, lon) dos passageiros.

Principais componentes
----------------------
* ``EAVRPGroup`` – extensão de ``Group`` que oferece centróides e distância
  interna ótima.
* ``GeoGrouper``  – algoritmo Greedy Capacity‑Constrained Clustering (G‑CCC).
"""

from __future__ import annotations

import itertools
import math
from typing import List, Sequence, Tuple

from utils import Passenger, Group  # classes fornecidas pelo usuário
from geografics import Distances

Coord = Tuple[float, float]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _centroid(points: Sequence[Coord]) -> Coord:
    """Retorna o centróide (lat, lon) de um conjunto de coordenadas."""
    lat = sum(p[0] for p in points) / len(points)
    lon = sum(p[1] for p in points) / len(points)
    return (lat, lon)


def _avg_intercluster_distance(cluster_a: Sequence[int], cluster_b: Sequence[int],
                               dist_matrix: List[List[float]]) -> float:
    """Média das distâncias *pré‑computadas* entre dois clusters (índices)."""
    total = 0.0
    for i in cluster_a:
        for j in cluster_b:
            total += dist_matrix[i][j]
    return total / (len(cluster_a) * len(cluster_b))


# ---------------------------------------------------------------------------
# Extensão de Group
# ---------------------------------------------------------------------------

class EAVRPGroup(Group):
    """Grupo de passageiros com utilidades extras (centróide & rota interna)."""

    # ---------------------------------------------------------------------
    # Centróides
    # ---------------------------------------------------------------------
    def origin_centroid(self) -> Coord:
        return _centroid([p.origin for p in self.passengers])

    def destination_centroid(self) -> Coord:
        return _centroid([p.destination for p in self.passengers])

    # ---------------------------------------------------------------------
    # Distância interna ótima (TSP exaustivo, k ≤ 5)
    # ---------------------------------------------------------------------
    def internal_distance(self, metric: str = "euclidean") -> float:
        metric_fn = getattr(Distances, metric)
        start = self.origin_centroid()
        dests = [p.destination for p in self.passengers]
        best = math.inf

        for perm in itertools.permutations(dests):
            length = 0.0
            pos = start
            for d in perm:
                # Distances.<metric>(x1, y1, x2, y2)
                length += metric_fn(*pos, *d)
                pos = d
            best = min(best, length)
        return best

    # Representação amigável para debug
    def __repr__(self) -> str:  # noqa: D401
        ids = [p.id for p in self.passengers]
        return f"<Group {self.id} – passengers {ids}>"


# ---------------------------------------------------------------------------
# Algoritmo de agrupamento
# ---------------------------------------------------------------------------

class GeoGrouper:
    """Agrupa passageiros em clusters de até 5 usando G‑CCC.

    Parameters
    ----------
    max_size : int, default=5
        Capacidade máxima de passageiros por grupo.
    alpha, beta : float, default=0.5
        Pesos da distância de origem e destino, respectivamente.
    penalty : float, default=10_000.0
        Penalidade aplicada a grupos incompletos
    metric : str, default="euclidean"
        Nome do método da classe ``Distances`` a empregar.
    """

    def __init__(self,
                 max_size: int = 5,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 penalty: float = 10_000.0,
                 metric: str = "euclidean") -> None:
        if not 1 <= max_size <= 5:
            raise ValueError("max_size precisa estar entre 1 e 5")
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.penalty = penalty
        self.metric_fn = getattr(Distances, metric)

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------
    def fit(self, passengers: Sequence[Passenger]) -> List[EAVRPGroup]:
        if not passengers:
            return []

        # 1) Matriz de distâncias combinadas -----------------------------
        n = len(passengers)
        dmat: List[List[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = self._pair_distance(passengers[i], passengers[j])
                dmat[i][j] = dmat[j][i] = d

        # 2) Inicializa clusters singleton ------------------------------
        clusters: List[set[int]] = [{i} for i in range(n)]

        # 3) Fusões greedy ----------------------------------------------
        while True:
            best_cost = math.inf
            best_pair: Tuple[int, int] | None = None

            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    new_size = len(clusters[a]) + len(clusters[b])
                    if new_size > self.max_size:
                        continue
                    inter = _avg_intercluster_distance(clusters[a], clusters[b], dmat)
                    cost = (self.max_size - new_size) * self.penalty + inter
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (a, b)

            if best_pair is None:
                break

            a, b = best_pair
            clusters[a].update(clusters[b])
            clusters.pop(b)

        # 4) Materializa objetos EAVRPGroup ------------------------------
        groups: List[EAVRPGroup] = []
        for gid, idx_set in enumerate(clusters, start=1):
            pax = [passengers[i] for i in sorted(idx_set)]
            groups.append(EAVRPGroup(gid, pax))
        return groups

    # ------------------------------------------------------------------
    # Auxiliar (distância composta)
    # ------------------------------------------------------------------
    def _pair_distance(self, p_i: Passenger, p_j: Passenger) -> float:
        d_orig = self.metric_fn(*p_i.origin, *p_j.origin)
        d_dest = self.metric_fn(*p_i.destination, *p_j.destination)
        return self.alpha * d_orig + self.beta * d_dest


# ---------------------------------------------------------------------------
# Conveniência de exportação
# ---------------------------------------------------------------------------
__all__ = ["GeoGrouper", "EAVRPGroup"]
