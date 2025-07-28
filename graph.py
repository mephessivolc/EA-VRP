import networkx as nx
import matplotlib.pyplot as plt
import random
from geografics import Distances

class GraphBuilder:
    def __init__(self, passengers, recharge_points, grouped_passengers=None, distance_func=Distances.euclidean):
        self.passengers = passengers
        self.grouped_passengers = grouped_passengers
        self.recharge_points = recharge_points
        self.distance_func = distance_func
        self.graph = nx.Graph()
        self.positions = {}

    def _add_passenger_nodes(self):
        for p in self.passengers:
            if not p in self.grouped_passengers:
                oid = f"PO{p.id}"
                did = f"PD{p.id}"
                self.graph.add_node(oid, pos=tuple(p.origin), tipo="origin")
                self.graph.add_node(did, pos=tuple(p.destination), tipo="destination")
                self.positions[oid] = tuple(p.origin)
                self.positions[did] = tuple(p.destination)
                self._add_edge_with_weight(oid, did)

    def _add_grouped_passenger_nodes(self):
        for p in self.grouped_passengers:
            oid = f"GO{p.id}"
            did = f"GD{p.id}"
            self.graph.add_node(oid, pos=tuple(p.origin), tipo="group_origin")
            self.graph.add_node(did, pos=tuple(p.destination), tipo="group_destination")
            self.positions[oid] = tuple(p.origin)
            self.positions[did] = tuple(p.destination)
            self._add_edge_with_weight(oid, did)    

    def _add_recharge_nodes(self):
        for r in self.recharge_points:
            r_id = f"R{r.id}"
            self.graph.add_node(r_id, pos=tuple(r.location), tipo="recharge")
            self.positions[r_id] = tuple(r.location)
            for other in self.graph.nodes:
                if other != r_id:
                    self._add_edge_with_weight(r_id, other)

    def _add_extra_edges(self, count=5):
        all_nodes = list(self.graph.nodes)
        for _ in range(count):
            n1, n2 = random.sample(all_nodes, 2)
            if not self.graph.has_edge(n1, n2):
                self._add_edge_with_weight(n1, n2)

    def _add_edge_with_weight(self, a, b):
        pos_a = self.positions[a]
        pos_b = self.positions[b]
        weight = self.distance_func(pos_a[0], pos_a[1], pos_b[0], pos_b[1])
        self.graph.add_edge(a, b, weight=weight)

    def build(self):
        self._add_passenger_nodes()
        self._add_grouped_passenger_nodes()
        self._add_recharge_nodes()
        self._add_extra_edges()

    def draw(self, filename=None):
        plt.figure(figsize=(12, 8))
        pos = {k: tuple(map(float, v[:2])) for k, v in self.positions.items()}
        labels = {n: n for n in self.graph.nodes}

        node_colors = []
        font_colors = []
        for n in self.graph.nodes:
            tipo = self.graph.nodes[n].get("tipo")
            if tipo == "group_origin":
                node_colors.append("green")
                font_colors.append("black")
            elif tipo == "group_destination":
                node_colors.append("blue")
                font_colors.append("white")
            elif tipo == "origin":
                node_colors.append("lightgreen")
                font_colors.append("black")
            elif tipo == "destination":
                node_colors.append("lightblue")
                font_colors.append("black")
            elif tipo == "recharge":
                node_colors.append("orange")
                font_colors.append("black")
            else:
                node_colors.append("lightgray")
                font_colors.append("black")

        nx.draw(self.graph, pos, with_labels=False, node_color=node_colors, node_size=700)
        for (node, (x, y), label, font_color) in zip(self.graph.nodes, pos.values(), labels.values(), font_colors):
            plt.text(x, y, label, fontsize=8, ha='center', va='center', color=font_color)

        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        formatted_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
        try:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=formatted_labels, font_size=6, rotate=False)
        except ValueError as e:
            print("Erro ao desenhar rótulos das arestas:", e)

        plt.title("Grafo de Passageiros e Grupos")
        if filename:
            plt.savefig(filename, dpi=300)
            print(f"Grafo {filename} criado com sucesso!!!")
        plt.show()

if __name__ == "__main__":
    from utils import Passenger, RechargePoint
    from geografics import generate_random_geografic_points

    print("--- Verificando construção do grafo ---")
    n = 5
    origens = generate_random_geografic_points(n)
    destinos = generate_random_geografic_points(n)
    passageiros = [Passenger(i, o, d) for i, (o, d) in enumerate(zip(origens, destinos))]

    pontos = generate_random_geografic_points(2)
    recargas = [RechargePoint(i, loc) for i, loc in enumerate(pontos)]

    g = GraphBuilder(passengers=passageiros, recharge_points=recargas, grouped_passengers=[passageiros])
    g.build()
    g.draw("figs/grafo_exemplo.png")
