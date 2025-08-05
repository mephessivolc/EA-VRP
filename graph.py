"""graph_builder.py
====================
Cria e salva um grafo EA-VRP usando **NetworkX**.

Elementos representados
----------------------
* **Origem passageiro**   – rótulo ``O{id}``  cor *lightblue*
* **Destino passageiro**  – rótulo ``D{id}``  cor *lightgreen*
* **Origem do grupo**     – rótulo ``GO{id}`` cor *blue*
* **Destino do grupo**    – rótulo ``GD{id}`` cor *green*
* **Ponto de recarga**    – rótulo ``R{id}``  cor *red*

Arestas
~~~~~~~
Cada par origem→destino recebe uma aresta com rótulo igual à distância
Euclidiana (ou Haversine, se preferir) obtida em
``geografics.Distances``.

Função principal
~~~~~~~~~~~~~~~~
``build_eavrp_graph(passengers, groups, stations, out_png, metric='euclidean')``

* **passengers** – lista de ``Passenger``
* **groups** – lista de ``EAVRPGroup``
* **stations** – lista de tuplas ``(lat, lon)`` **ou** objetos com atributo
  coordenada (``coord``, ``location``, ``pos`` ou similar)
* **out_png** – caminho do arquivo de saída
* **metric** – 'euclidean' ou outro método de ``Distances``
"""

from __future__ import annotations

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx

from geografics import Distances
from grouping import EAVRPGroup
from utils import Passenger

Coord = Tuple[float, float]
Station = Union[Coord, object]


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def _extract_xy(obj: Station) -> Coord:
    """Devolve (x, y) de `obj`. Aceita tupla/lista ou objeto com atributo."""
    if isinstance(obj, (tuple, list)):
        return tuple(obj[:2])  # type: ignore[arg-type]
    # tenta atributos comuns
    for attr in ("coord", "coords", "location", "pos", "position"):
        xy = getattr(obj, attr, None)
        if xy is not None:
            return tuple(xy[:2])  # type: ignore[arg-type]
    raise TypeError("Station não possui coordenadas acessíveis")


def build_graph(passengers: List[Passenger],
                      groups: List[EAVRPGroup],
                      stations: List[Station],
                      out_png: str | None,
                      metric: str = "euclidean") -> None:
    """Gera o grafo EA-VRP e salva em *out_png* (PNG)."""
    metric_fn = getattr(Distances, metric)

    G = nx.Graph()

    # ---------------- Passenger nodes & edges ----------------------
    for p in passengers:
        o_node = f"O{p.id}"
        d_node = f"D{p.id}"
        G.add_node(o_node, pos=tuple(p.origin()[:2]), color="lightblue", label=o_node)
        G.add_node(d_node, pos=tuple(p.destination()[:2]), color="lightgreen", label=d_node)
        dist = metric_fn(*p.origin(), *p.destination())
        G.add_edge(o_node, d_node, weight=dist)

    # ---------------- Group centroid nodes & edges ----------------
    for g in groups:
        go_node = f"GO{g.id}"
        gd_node = f"GD{g.id}"
        G.add_node(go_node, pos=tuple(g.origin()[:2]), color="blue", label=go_node)
        G.add_node(gd_node, pos=tuple(g.destination()[:2]), color="green", label=gd_node)
        dist = metric_fn(*g.origin(), *g.destination())
        G.add_edge(go_node, gd_node, weight=dist)

    # ---------------- Charging stations ---------------------------
    for idx, st in enumerate(stations, start=1):
        xy = _extract_xy(st)
        r_node = f"R{idx}"
        G.add_node(r_node, pos=xy, color="red", label=r_node)

    station_nodes = [n for n in G.nodes if n.startswith("R")]
    for r_node in station_nodes:
        r_pos = G.nodes[r_node]["pos"]
        for n, data in G.nodes(data=True):
            if n.startswith("R"):
                continue            # pula arestas R↔R
            dist = metric_fn(*r_pos, *data["pos"])
            G.add_edge(r_node, n, weight=dist)
    # ---------------- Drawing -------------------------------------
    pos = nx.get_node_attributes(G, "pos")
    colors = [data["color"] for _, data in G.nodes(data=True)]
    labels = nx.get_node_attributes(G, "label")
    edge_labels = {e: f"{d['weight']:.1f}" for e, d in G.edges.items() if "weight" in d}

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, node_color=colors, with_labels=True, labels=labels,
            node_size=500, font_size=8, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
    plt.show()
    plt.close()


__all__ = ["build_graph"]