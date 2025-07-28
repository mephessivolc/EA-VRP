"""quantum.py – Geração do QUBO usando Assignment
================================================
A classe **QuboGenerator** agora recebe um objeto **Assignment**, que já contém a
vinculação veículo → grupo (priorizando grupos maiores). Cada variável binária do
QUBO segue o formato

    V<vehicle_id>G<group_id>

representando a decisão “o veículo V<i> realiza a rota do grupo G<j> (origem →
destino)”. Somente grupos atribuídos a veículos compõem o QUBO.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from geografics import Distances
from grouper import Assignment
from utils import Vehicle, RechargePoint, Passenger


class QuboGenerator:
    """Constrói a matriz QUBO para o EA‑VRP considerando apenas grupos atribuídos.

    Parâmetros
    ----------
    assignment : Assignment
        Objeto contendo o mapeamento veículo → grupo.
    recharge_points : list[RechargePoint]
        Pontos de recarga disponíveis.
    distance_func : callable, opcional
        Função de distância (default: Distances.euclidean).
    """

    def __init__(self, assignment: Assignment, recharge_points: List[RechargePoint],
                 distance_func=Distances.euclidean):
        self.assignment = assignment
        self.recharge_points = recharge_points
        self.distance_func = distance_func

        self.variables: List[str] = []  # ex.: ["V0G7", "V1G3", ...]
        self.qubo: Dict[Tuple[str, str], float] = defaultdict(float)

    # ---------------------------------------------------------------------
    # Construção das variáveis
    # ---------------------------------------------------------------------
    def build_variables(self) -> None:
        """Cria a lista de variáveis binárias baseado no Assignment."""
        self.variables = []
        for v_key, group in self.assignment.mapping.items():
            g_key = f"G{group[0].id}"
            self.variables.append(f"{v_key}{g_key}")

    # ---------------------------------------------------------------------
    # Termo de custo (distância percorrida) – linear
    # ---------------------------------------------------------------------
    def add_travel_costs(self) -> None:
        """Custo proporcional à distância percurso origem→destino de cada grupo."""
        for v_key, group in self.assignment.mapping.items():
            vehicle: Vehicle = self._vehicle_from_key(v_key)
            origin = group[0].origin
            destination = group[0].destination
            dist = self.distance_func(origin[0], origin[1], destination[0], destination[1])
            cost = dist * vehicle.consumption_per_km

            var = f"{v_key}G{group[0].id}"
            self.qubo[(var, var)] += cost

    # ---------------------------------------------------------------------
    # Restrição de energia – penalidade
    # ---------------------------------------------------------------------
    def add_energy_constraints(self) -> None:
        """Penaliza rotas inviáveis energeticamente (entrega + recarga)."""
        for v_key, group in self.assignment.mapping.items():
            vehicle = self._vehicle_from_key(v_key)
            origin = group[0].origin
            destination = group[0].destination

            # Energia para entregar o grupo
            to_dest = self.distance_func(origin[0], origin[1], destination[0], destination[1])
            energy_to_dest = vehicle.energy_needed(to_dest)

            # Energia para chegar ao ponto de recarga mais próximo após entrega
            r_dists = [self.distance_func(destination[0], destination[1], r.location[0], r.location[1])
                       for r in self.recharge_points]
            to_recharge = min(r_dists)
            energy_to_recharge = vehicle.energy_needed(to_recharge)

            total_energy = energy_to_dest + energy_to_recharge

            if vehicle.battery - total_energy < vehicle.min_charge:
                var = f"{v_key}G{group[0].id}"
                self.qubo[(var, var)] += 1000  # penalidade alta

    # ---------------------------------------------------------------------
    # Construção completa do QUBO
    # ---------------------------------------------------------------------
    def build_qubo(self) -> Dict[Tuple[str, str], float]:
        self.build_variables()
        self.add_travel_costs()
        self.add_energy_constraints()
        return self.qubo

    # ---------------------------------------------------------------------
    # Utilidades
    # ---------------------------------------------------------------------
    def get_variable_index_map(self) -> Dict[str, int]:
        return {var: idx for idx, var in enumerate(self.variables)}

    def export_matrix(self) -> np.ndarray:
        idx = self.get_variable_index_map()
        size = len(idx)
        Q = np.zeros((size, size))
        for (i, j), val in self.qubo.items():
            Q[idx[i], idx[j]] += val
        return Q

    def _vehicle_from_key(self, v_key: str) -> Vehicle:
        """Helper para recuperar objeto Vehicle a partir de "V<id>"."""
        vid = int(v_key[1:])
        return next(v for v in self.assignment.vehicles if v.id == vid)


# ---------------------------------------------------------------------
# Teste rápido
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from utils import Passenger, Vehicle, RechargePoint
    from geografics import generate_random_geografic_points
    from grouper import PassengerGrouper, Assignment

    # ---------------- Dados sintéticos ---------------- #
    n_passengers = 10
    n_vehicles = 3
    n_recharges = 2
    eps = 0.3

    origins = generate_random_geografic_points(n_passengers, (-1, 1), (-1, 1))
    destinations = generate_random_geografic_points(n_passengers, (-1, 1), (-1, 1))
    passengers = [Passenger(i, o, d) for i, (o, d) in enumerate(zip(origins, destinations))]

    vehicles = [Vehicle(i, battery=40.0) for i in range(n_vehicles)]
    recharge_points = [RechargePoint(i, loc) for i, loc in enumerate(generate_random_geografic_points(n_recharges, (-1, 1), (-1, 1)))]

    # ---------------- Agrupamento e atribuição ---------------- #
    grouper = PassengerGrouper(passengers, eps=eps)
    groups = grouper.build_groups()
    assign = Assignment(vehicles, groups)

    print("Assignment string:", str(assign))

    # ---------------- Geração do QUBO ---------------- #
    qgen = QuboGenerator(assign, recharge_points)
    Q_dict = qgen.build_qubo()
    Q = qgen.export_matrix()

    print("Variáveis:", qgen.variables)
    print("Matriz QUBO densa:\n", Q)
