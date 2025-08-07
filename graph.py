"""graph_builder.py
====================
Cria e salva um grafo EA-VRP usando **NetworkX**.

Elementos representados
----------------------
* **Origem passageiro**   - rótulo ``O{id}``  cor *lightblue*
* **Destino passageiro**  - rótulo ``D{id}``  cor *lightgreen*
* **Origem do grupo**     - rótulo ``GO{id}`` cor *blue*
* **Destino do grupo**    - rótulo ``GD{id}`` cor *green*
* **Ponto de recarga**    - rótulo ``R{id}``  cor *red*

Arestas
~~~~~~~
Cada par origem→destino recebe uma aresta com rótulo igual à distância
Euclidiana (ou Haversine, se preferir) obtida em
``geografics.Distances``.

Função principal
~~~~~~~~~~~~~~~~
``build_eavrp_graph(passengers, groups, stations, out_png, metric='euclidean')``

* **passengers** - lista de ``Passenger``
* **groups** - lista de ``EAVRPGroup``
* **stations** - lista de tuplas ``(lat, lon)`` **ou** objetos com atributo
  coordenada (``coord``, ``location``, ``pos`` ou similar)
* **out_png** - caminho do arquivo de saída
* **metric** - 'euclidean' ou outro método de ``Distances``
"""

from __future__ import annotations

from typing import List, Tuple, Union

import matplotlib.pyplot as plt # type: ignore
import networkx as nx # type: ignore

from geografics import Distances
from grouping import EAVRPGroup
from utils import Passenger, Depot, RechargePoint

Coord = Tuple[float, float]
Station = Union[Coord, object]


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def calc_distance(fn, o: Tuple[float,float], d: Tuple[float,float], dim=100) -> float:
    return fn(*o, *d) * dim

def build_graph(
        passengers: List[Passenger], 
        groups: List[EAVRPGroup], 
        recharge_points: List[RechargePoint],
        depots: List[Depot],
        out_png: str | None,
        metric: str = "euclidean") -> None:
    
    """Gera o grafo EA-VRP e salva em *out_png* (PNG)."""
    metric_fn = getattr(Distances, metric)

    G = nx.Graph()

    # ---------------- Passenger nodes & edges ----------------------
    for p in passengers:
        o_node = p.str_origin
        d_node = p.str_destination
        G.add_node(o_node, pos=tuple(p.origin), color="lightblue", label=o_node)
        G.add_node(d_node, pos=tuple(p.destination), color="lightgreen", label=d_node)
        dist = calc_distance(metric_fn, p.origin, p.destination)
        
        G.add_edge(o_node, d_node, weight=dist)

    # ---------------- Group centroid nodes & edges ----------------
    for g in groups:
        go_node = g.str_origin
        gd_node = g.str_destination
        G.add_node(go_node, pos=tuple(g.origin), color="blue", label=go_node)
        G.add_node(gd_node, pos=tuple(g.destination), color="green", label=gd_node)
        dist = calc_distance(metric_fn, g.origin, g.destination)
        G.add_edge(go_node, gd_node, weight=dist)

    # ---------------- Charging stations ---------------------------
    for r in recharge_points:
        r_node = r.str_location
        G.add_node(r_node, pos=tuple(r.location), color="red", label=r_node)

    station_nodes = [n for n in G.nodes if n.startswith("R")]
    for r_node in station_nodes:
        r_pos = G.nodes[r_node]["pos"]
        for n, data in G.nodes(data=True):
            if n.startswith("R"):
                continue            # pula arestas R-R
            # dist = metric_fn(*r_pos, *data["pos"])
            print(r_pos)
            dist = calc_distance(metric_fn, r_pos, data["pos"])
            G.add_edge(r_node, n, weight=dist)

    # ---------------- Depot stations ---------------------------
    for d in depots:
        g_node = d.str_location
        G.add_node(g_node, pos=d.location, color='yellow', label=g_node)

    depot_nodes = [n for n in G.nodes if n.startswith("D")]
    for d_node in depot_nodes:
        d_pos = G.nodes[d_node]["pos"]
        for n, data in G.nodes(data=True):
            if n.startswith("D"): # pula arestas D-D
                continue 
            dist = calc_distance(metric_fn, d_pos, data["pos"])
            G.add_edge(d_node, n, weight=dist)

    # ---------------- Drawing -------------------------------------
    pos = nx.get_node_attributes(G, "pos")
    colors = [data["color"] for _, data in G.nodes(data=True)]
    labels = nx.get_node_attributes(G, "label")
    edge_labels = {e: f"{d['weight']:.1f}" for e, d in G.edges.items() if "weight" in d}

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_color=colors, with_labels=True, labels=labels,
            node_size=500, font_size=8, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    # ----------- Escala (canto inferior esquerdo) -----------------
    plt.gca().text(
        0.02,  # x – 2% da largura da área de desenho
        0.02,  # y – 2% da altura da área de desenho
        "Escala 1:100",
        transform=plt.gca().transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5),
    )

    plt.axis("off")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=300)
    plt.show()
    plt.close()


__all__ = ["build_graph"]