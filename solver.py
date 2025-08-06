"""qaoa_solver.py
=================
QAOA solver para o QUBO de EA‑VRP (compatível com PennyLane ≤ 0.37).

Novidades
---------
* **save_histogram** colore a barra do **melhor estado** em destaque
  (vermelho) e mantém rótulos em **10** posições igualmente espaçadas.
* Corrigido erro tipográfico na assinatura de `interpret`.
"""

from __future__ import annotations

from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

from pennylane import ApproxTimeEvolution

from encoder import QUBOEncoder


class QAOASolver:
    """QAOA solver com utilidades de análise e histograma."""

    def __init__(self,
                 encoder: QUBOEncoder,
                 p: int = 1,
                 shots: int = 1_000,
                 steps: int = 100,
                 lr: float = 0.1,
                 trotter_steps: int = 1,
                 dev: str = "lightning.qubit",
                 seed: int | None = None) -> None:
        self.encoder = encoder
        self.p = p
        self.shots = shots
        self.steps = steps
        self.seed = seed
        self.n_qubits = encoder.num_qubits
        self.trotter_steps = trotter_steps

        # Dispositivo quântico
        try:
            self.dev = qml.device(dev, wires=self.n_qubits, shots=shots, seed=seed)
        except Exception as e:
            raise ValueError(f"Pennylane não conseguiu instanciar o device '{dev}': {e}")

        # Hamiltoniano de custo
        self.cost_h, self.offset = self._qubo_to_hamiltonian()

        # Parâmetros iniciais QAOA
        rng = np.random.default_rng(seed)
        self.params = 0.01 * rng.standard_normal(size=(2, p))
        self.opt = qml.AdamOptimizer(lr)

        # Resultados
        self.history: List[float] = []
        self.samples: np.ndarray | None = None
        self.best_bits: List[int] | None = None
        self.best_cost: float | None = None

    # ---------------------------------------------------------------
    # Construção do Hamiltoniano
    # ---------------------------------------------------------------
    def _qubo_to_hamiltonian(self):
        coeffs, ops = [], []
        offset = self.encoder.offset
        for (i, j), q in self.encoder.Q.items():
            if i == j:
                offset += q / 2
                coeffs.append(-q / 2)
                ops.append(qml.PauliZ(i))
            else:
                offset += q / 4
                coeffs.extend([-q / 4, -q / 4, q / 4])
                ops.extend([qml.PauliZ(i), qml.PauliZ(j), qml.PauliZ(i) @ qml.PauliZ(j)])
        return qml.Hamiltonian(coeffs, ops), offset

    # ---------------------------------------------------------------
    # Camadas do circuito QAOA
    # ---------------------------------------------------------------
    def _apply_cost(self, gamma: float):
        # qml.apply(qml.evolve(self.cost_h, gamma))
        qml.apply(
            ApproxTimeEvolution(self.cost_h, gamma, self.trotter_steps)
        )

    def _apply_mixer(self, beta: float):
        for w in range(self.n_qubits):
            qml.RX(2 * beta, wires=w)

    # ---------------------------------------------------------------
    # Circuitos auxiliares
    # ---------------------------------------------------------------
    def _expval_circuit(self, params):
        gammas, betas = params
        for w in range(self.n_qubits):
            qml.Hadamard(wires=w)
        for l in range(self.p):
            self._apply_cost(gammas[l])
            self._apply_mixer(betas[l])
        return qml.expval(self.cost_h)

    def _sample_circuit(self, params):
        gammas, betas = params
        for w in range(self.n_qubits):
            qml.Hadamard(wires=w)
        for l in range(self.p):
            self._apply_cost(gammas[l])
            self._apply_mixer(betas[l])
        return [qml.sample(qml.PauliZ(wires=w)) for w in range(self.n_qubits)]

    # ---------------------------------------------------------------
    # Fase de otimização
    # ---------------------------------------------------------------
    def solve(self):
        cost_qnode = qml.QNode(self._expval_circuit, self.dev)
        params = self.params.copy()
        for _ in range(self.steps):
            params, cost = self.opt.step_and_cost(cost_qnode, params)
            self.history.append(cost + self.offset)

        # Amostragem final
        sample_qnode = qml.QNode(self._sample_circuit, self.dev)
        raw = np.array(sample_qnode(params)).T  # (shots, n_qubits)
        bits = ((1 - raw) // 2).astype(int)
        self.samples = bits

        # Avalia custo de cada bitstring
        costs = [self._bit_cost(b) for b in bits]
        best_idx = int(np.argmin(costs))
        self.best_bits = bits[best_idx].tolist()
        self.best_cost = float(costs[best_idx])
        return self.best_bits, self.best_cost

    # ---------------------------------------------------------------
    def _bit_cost(self, bits):
        y = bits.astype(float)
        return y @ self.encoder.to_matrix() @ y + self.encoder.offset

    # ---------------------------------------------------------------
    # Histograma
    # ---------------------------------------------------------------
    def plot_histogram(self, filename: str | None):
        """Plota histograma de todos os estados com **10 rótulos**.

        A barra correspondente ao melhor estado é destacada em vermelho.
        """
        if self.samples is None:
            raise RuntimeError("Execute solve() antes.")

        # Converte bitstrings → inteiros
        xs = [int("".join(map(str, b[::-1])), 2) for b in self.samples]
        uniq, counts = np.unique(xs, return_counts=True)

        # Índices para rótulos igualmente espaçados
        n_states = len(uniq)
        label_positions = np.linspace(0, n_states - 1, min(10, n_states), dtype=int)
        label_positions_set = set(label_positions)
        labels = [f"|{u}⟩" if idx in label_positions_set else "" for idx, u in enumerate(uniq)]

        # Cores: destaque para melhor estado
        best_int = int("".join(map(str, self.best_bits[::-1])), 2) if self.best_bits else None
        colors = ["tab:red" if u == best_int else "tab:blue" for u in uniq]

        plt.figure(figsize=(12, 8))
        plt.bar(range(n_states), counts, color=colors)
        plt.xticks(range(n_states), labels, rotation=45, ha="right")
        plt.xlabel("Estados |x⟩")
        plt.ylabel("Frequência")
        plt.title("Histograma QAOA")
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
        plt.show()
        plt.close()

    # ---------------------------------------------------------------
    # Interpretação do bitstring
    # ---------------------------------------------------------------
    def interpret(self, bitstring: List[int]):
        mapping = {}
        for idx, val in enumerate(bitstring):
            if val == 1:
                v, g = self.encoder.reverse_map[idx]
                mapping.setdefault(v, []).append(g)
        return mapping


__all__ = ["QAOASolver"]
