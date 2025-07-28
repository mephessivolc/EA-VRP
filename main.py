"""main.py – Pipeline completa
=============================
Geração de dados → agrupamento → atribuição → grafo → QUBO.
"""

from utils import Passenger, Vehicle, RechargePoint
from geografics import generate_random_geografic_points
from graph import GraphBuilder
from grouper import PassengerGrouper, Assignment
from quantum import QuboGenerator
from optimizer import QuantumRouteOptimizer

# ------------------------------- Parâmetros ------------------------------- #

n_passengers = 20
n_vehicles = 15
n_recharge_points = 8
eps = 0.3

lat_range = (-23.9, -23.3)
lon_range = (-46.8, -46.4)

# ------------------------------ Geração Dados ----------------------------- #

origins = generate_random_geografic_points(n_passengers, lat_range, lon_range)
destinations = generate_random_geografic_points(n_passengers, lat_range, lon_range)
recharge_coords = generate_random_geografic_points(n_recharge_points, lat_range, lon_range)

passengers = [Passenger(i, o, d) for i, (o, d) in enumerate(zip(origins, destinations))]
vehicles = [Vehicle(i) for i in range(n_vehicles)]
recharge_points = [RechargePoint(i, loc) for i, loc in enumerate(recharge_coords)]

# --------------------------- Agrupamento & Atrib -------------------------- #

grouper = PassengerGrouper(passengers, eps=eps)
groups = grouper.build_groups()
assignment = Assignment(groups, vehicles)

print("Assignment:", assignment.print_report())

# ------------------------------- Construir Grafo -------------------------- #

graph_builder = GraphBuilder(passengers=passengers,
                             recharge_points=recharge_points,
                             grouped_passengers=assignment.as_list())

graph_builder.build()
graph_builder.draw(f"figs/Graph_P{n_passengers}V{n_vehicles}R{n_recharge_points}.png")

# --------------------------- Gerar QUBO & matriz -------------------------- #

qgen = QuboGenerator(assignment, recharge_points)
qubo = qgen.build_qubo()
Q = qgen.export_matrix()

print("Variáveis do QUBO:", qgen.variables)
print("Matriz QUBO densa:\n", Q)

print(f"Quantidade de Qubits utilizado: {len(qgen.variables)}")

optimizer = QuantumRouteOptimizer(qubo, qgen.variables)
exp_cost = optimizer.solve_qaoa(p=2, steps=5, lr=0.1, shots=1000)

print(f"Valor esperado: {exp_cost}")
print(f"Melhor estado: {optimizer.best_bitstring}")

filename = f"figs/histogram_P{n_passengers}V{n_vehicles}R{n_recharge_points}.png"
optimizer.plot_histogram(filename=filename)
