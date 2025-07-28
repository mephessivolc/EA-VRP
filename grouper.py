import random
from itertools import combinations
from typing import List, Dict, Tuple
from utils import Passenger, Vehicle

from geografics import Distances

class PassengerGrouper:
    """Agrupa passageiros em blocos de 1 até **max_group_size** pessoas
    priorizando grupos maiores e respeitando uma distância máxima *eps*.
    """

    def __init__(self, 
                 passengers, 
                 distance_func=Distances.euclidean,
                 max_group_size: int = 5, 
                 eps: float = 0.2
                 ):
        
        self.passengers = passengers[:]
        self.distance_func = distance_func
        self.max_group_size = max_group_size
        self.eps = eps  # distância máxima entre passageiros no mesmo grupo
        self.groups: List[List["Passenger"]] = []

    # ------------------------------ Métodos internos ----------------------------- #

    def _distance(self, p1, p2) -> float:
        return self.distance_func(p1.origin[0], p1.origin[1], p2.origin[0], p2.origin[1])

    def _pairwise_within_eps(self, group: Tuple["Passenger", ...]) -> bool:
        """Verifica se todos os pares do grupo estão dentro de *eps*."""
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if self._distance(group[i], group[j]) > self.eps:
                    return False
        return True

    def _greedy_grouping(self):
        """Cria grupos priorizando tamanho 5 → 4 → 3 → 2 → 1."""
        remaining = self.passengers[:]
        used_ids = set()

        for size in range(self.max_group_size, 0, -1):
            for combo in combinations(remaining, size):
                if any(p.id in used_ids for p in combo):
                    continue
                if self._pairwise_within_eps(combo):
                    self.groups.append(list(combo))
                    used_ids.update(p.id for p in combo)
            remaining = [p for p in remaining if p.id not in used_ids]

    # ----------------------------- API pública ----------------------------- #

    def build_groups(self) -> List[List["Passenger"]]:
        """Constroi grupos e retorna lista ordenada por tamanho (desc)."""
        self.groups = []
        self._greedy_grouping()
        self.groups.sort(key=len, reverse=True)
        return self.groups

    def get_group_summary(self) -> List[Tuple[int, List[int]]]:
        return [(len(g), [p.id for p in g]) for g in self.groups]


class Assignment:
    """Associa grupos a veículos priorizando grupos maiores.
    - Mapeamento em self.mapping {"V<id>": grupo}
    - Grupos não atribuídos em self.unassigned
    """

    def __init__(self, 
                 groups: List[List[Passenger]], 
                 vehicles: List[Vehicle] | None=None
                 ):
        # se vehicles for None ou vazio, cria um Vehicle fictício para cada grupo
        if not vehicles:
            vehicles = [Vehicle(i) for i in range(len(groups))]
        
        self.vehicles = vehicles # sorted(vehicles, key=lambda v: v.id)
        self.groups = groups # sorted(groups, key=len, reverse=True)
        self.mapping: Dict[str, List[Passenger]] = {}
        self.unassigned: List[List[Passenger]] = []
        self._assign()

    def _assign(self):
        v_iter = iter(self.vehicles)
        for group in self.groups:
            try:
                vehicle = next(v_iter)
                self.mapping[f"V{vehicle.id}"] = group
            except StopIteration:
                self.unassigned.append(group)

    # ----------------------------- Utilidades ----------------------------- #

    def as_list(self) -> List["Passenger"]:
        return [grp[0] for grp in self.mapping.values()]

    def get_assigment_list(self) -> str:
        """Retorna string com pares no formato V<id>G<group_id> separados por espaço."""
        if not self.mapping:
            return "<no assignment>"
        parts = [f"{v_key}_G{grp[0].id}" for v_key, grp in self.mapping.items()]
        return parts #" ".join(parts)

    def print_report(self):
        print("--- Veículos e seus Grupos ---")
        for v, grp in self.mapping.items():
            ids = [p.id for p in grp]
            print(f"{v}: passageiros {ids} (tam={len(grp)})")
        if self.unassigned:
            print("\n--- Grupos não atribuídos ---")
            for i, grp in enumerate(self.unassigned, 1):
                ids = [p.id for p in grp]
                print(f"R{i}: passageiros {ids} (tam={len(grp)})")


# --------------------------- Teste rápido --------------------------- #
if __name__ == "__main__":
    from utils import Passenger, Vehicle
    from geografics import generate_random_geografic_points

    random.seed(42)

    n_passengers = 12
    n_vehicles = 3
    eps = 0.3

    origins = generate_random_geografic_points(n_passengers, (-1, 1), (-1, 1))
    destinations = generate_random_geografic_points(n_passengers, (-1, 1), (-1, 1))

    passengers = [Passenger(i, o, d) for i, (o, d) in enumerate(zip(origins, destinations))]
    vehicles = [Vehicle(i) for i in range(n_vehicles)]

    grouper = PassengerGrouper(passengers, eps=eps)
    groups = grouper.build_groups()
    assign = Assignment(groups, vehicles)

    print("Resumo dos Grupos:", grouper.get_group_summary())
    assign.print_report()
    print("String assignment:", str(assign.get_assigment_list()))
