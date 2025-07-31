"""main.py
===========
Pipeline de demonstração do EA‑VRP.

Gera dados sintéticos (passageiros, veículos, pontos de recarga), executa o
agrupamento geográfico e imprime no console a composição de cada grupo.

Uso
---
$ python main.py 

"""

from __future__ import annotations

import random
from typing import List

from geografics import generate_random_geografic_points
from grouping import GeoGrouper
from utils import Passenger, Vehicle, RechargePoint
from encoder import EA_VRP_QUBOEncoder
from solver import QAOASolver
from graph import build_eavrp_graph

from pathlib import Path
# ---------------------------------------------------------------------------
# Helpers de geração sintética
# ---------------------------------------------------------------------------

def make_passengers(n: int, x_lim: List, y_lim: List) -> List[Passenger]:
    """Cria *n* passageiros com origem e destino aleatórios."""
    origin = generate_random_geografic_points(n, x_lim, y_lim)
    dest = generate_random_geografic_points(n, x_lim, y_lim)
    passengers: List[Passenger] = []
    for i, (o, d) in enumerate(zip(origin,dest)):
        passengers.append(Passenger(id=i, origin=o, destination=d))
    return passengers


def make_vehicles(n: int) -> List[Vehicle]:
    """Cria *n* veículos com parâmetros padrão."""
    return [Vehicle(id=i + 1) for i in range(n)]

def make_recharge_points(n: int, x_lim: List, y_lim: List) -> List[RechargePoint]:
    origin = generate_random_geografic_points(n, x_lim, y_lim)
    recharge_points: List[RechargePoint] = []
    for i, o in enumerate(origin):
        recharge_points.append(RechargePoint(id=i, location=o))
    return recharge_points

# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------

def main() -> None:
    OUTDIR = Path("figs")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    n_passengers = 10
    n_vehicles = 3
    n_recharge_points = 2

    p=2
    shots=1000
    steps=15

    seed = 42
    filename_hist = OUTDIR / "teste_histograma.png" # nome do arquivo do histograma
    filename_graph = OUTDIR / "teste_grafo.png" # nome do arquivo do grafo

    passengers = make_passengers(n_passengers, (-24, -23.4), (-46.8, -46.3))
    vehicles = make_vehicles(n_vehicles)
    recharge_points = make_recharge_points(n_recharge_points, (-24, -23.4), (-46.8, -46.3))

    grouper = GeoGrouper()
    groups = grouper.fit(passengers)

    # -----------------------------------------------------
    # Relatório simples no console
    # -----------------------------------------------------
    print("==== Agrupamento de Passageiros ====")
    for g in groups:
        p_ids = [passenger.id for passenger in g.passengers]
        print(f"Grupo {g.id:02d} | {len(p_ids)} passageiros -> {p_ids}")
    print("====================================")

    # Exemplo de como acessar métricas extras
    total_qubits = len(groups)  # variável por grupo (modelo futuro)
    print(f"\nTotal de grupos (≈ qubits na formulação): {total_qubits}")

    print("=== Total de Veículos ===")
    print(f"Veículos {len(vehicles)}")
    # 2) Construção do QUBO -------------------------------------------
    encoder = EA_VRP_QUBOEncoder(vehicles, groups, recharge_points)

    print(f"Variáveis / qubits: {encoder.num_qubits}")
    print("Matriz QUBO (valores arredondados a 2 casas decimais):")
    encoder.print_matrix(precision=2)

    # Exemplo: acessar o ndarray para outras análises -----------------
    mat = encoder.to_matrix()
    offset = encoder.offset
    print(f"\nOffset (constante): {offset:.2f}")

     # 3) QAOA ----------------------------------------------------------
    solver = QAOASolver(encoder,
                        p=p,
                        shots=shots,
                        steps=steps,
                        dev="default.mixed",
                        seed=seed)
    best_bits, best_cost = solver.solve()

    # Estado inteiro x (LSB = qubit 0)
    x_val = int("".join(map(str, best_bits[::-1])), 2)
    print(f"Melhor estado: |{x_val}⟩  ==> custo {best_cost:.2f}\n")

    # 4) Histograma ----------------------------------------------------
    
    solver.save_histogram(filename_hist)
    print(f"Histograma salvo em: {filename_hist}")

    build_eavrp_graph(passengers, groups, recharge_points, filename_graph)
    print(f"Grafo salvo em: {filename_graph}")

if __name__ == "__main__":
    main()
