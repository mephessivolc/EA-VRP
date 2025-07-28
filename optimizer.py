"""optimizer.py – QuantumRouteOptimizer (v5)
================================================

Resolve o EA‑VRP formulado como QUBO usando QAOA em PennyLane (simulação local)
e fornece opção de annealing com Ocean SDK.

Principais mudanças nesta versão:
* **Camadas QAOA atualizadas** – uso de `qml.qaoa.cost_layer` e
  `qml.qaoa.mixer_layer` com Hamiltonianos explícitos, compatíveis com
  PennyLane ≥ 0.30.
* **Parâmetros treináveis** inicializados com `pennylane.numpy` para garantir
  suporte a gradientes.
* **Bloco de teste** simplificado e corrigido.

Dependências:
```
pip install pennylane dimod matplotlib  # (+ dwave-ocean-sdk se desejar annealing)
```
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional

import pennylane as qml
from pennylane import numpy as np  # garante compatibilidade com autograd
import dimod

# Ocean SDK (opcional) para annealing ---------------------------------------
try:
    from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
except ModuleNotFoundError:  # sem Ocean instalado/credenciais
    DWaveSampler = EmbeddingComposite = LeapHybridSampler = None  # type: ignore


class QuantumRouteOptimizer:
    """Otimizador de rotas EA‑VRP via QAOA (local) ou annealing (opcional).

    Parâmetros
    ----------
    qubo : Dict[(str,str), float]
        Dicionário QUBO gerado anteriormente.
    variable_order : List[str]
        Lista de variáveis na ordem da bitstring.
    """

    def __init__(self, qubo: Dict[Tuple[str, str], float], variable_order: List[str]):
        self.qubo = qubo
        self.vars = variable_order
        self.var_index = {v: i for i, v in enumerate(self.vars)}
        self.best_bitstring: Optional[str] = None
        self.bitstring_counts: Optional[List[Tuple[str, int, float]]] = None

    # --------------------------- Conversões internas ----------------------- #
    def _qubo_indexed(self) -> Dict[Tuple[int, int], float]:
        """Converte QUBO (string) → QUBO (índices)."""
        return {(self.var_index[a], self.var_index[b]): w for (a, b), w in self.qubo.items()}

    def _ising_hamiltonian(self) -> qml.Hamiltonian:
        """Retorna Hamiltoniano Ising equivalente ao QUBO."""
        h, J, _ = dimod.qubo_to_ising(self._qubo_indexed())
        coeffs, obs = [], []
        for i, bias in h.items():
            if abs(bias) > 1e-12:
                coeffs.append(bias)
                obs.append(qml.PauliZ(i))
        for (i, j), coupl in J.items():
            if abs(coupl) > 1e-12:
                coeffs.append(coupl)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
        return qml.Hamiltonian(coeffs, obs)

    # --------------------------- QAOA (local) ------------------------------ #
    def solve_qaoa(
            self,
            p: int = 1,
            steps: int = 100,
            lr: float = 0.1,
            shots: Optional[int] = None,
        ) -> float:
        """Executa QAOA local e retorna valor esperado mínimo.

        Atualizado para usar camadas QAOA compatíveis e gradientes corretos.
        """
        n = len(self.vars)
        dev = qml.device("default.qubit", wires=n, shots=shots)
        cost_h = self._ising_hamiltonian()
        # Mixer padrão σ_x em todos os qubits
        mixer_h = qml.Hamiltonian([1.0] * n, [qml.PauliX(i) for i in range(n)])

        # --------------------- QNode de custo --------------------- #
        @qml.qnode(dev)
        def cost_qnode(params):
            # Inicia em |+>^n
            for w in range(n):
                qml.Hadamard(wires=w) #, pattern="single")
            for layer in range(p):
                gamma, beta = params[layer]
                qml.qaoa.cost_layer(gamma, cost_h)
                qml.qaoa.mixer_layer(beta, mixer_h)
            return qml.expval(cost_h)

        # --------------------- QNode de amostra -------------------- #
        @qml.qnode(dev)
        def sample_qnode(params):
            for w in range(n):
                qml.Hadamard(wires=w) #, pattern="single")
            for layer in range(p):
                gamma, beta = params[layer]
                qml.qaoa.cost_layer(gamma, cost_h)
                qml.qaoa.mixer_layer(beta, mixer_h)
            return qml.sample(wires=range(n))

        # Parâmetros treináveis (gamma, beta) iniciais
        params = 0.01 * np.random.randn(p, 2)
        opt = qml.AdamOptimizer(stepsize=lr)
        for _ in range(steps):
            params = opt.step(cost_qnode, params)
        exp_val = float(cost_qnode(params))

        # Amostragem estatística
        bit_samples = np.asarray(sample_qnode(params))
        if bit_samples.ndim == 1:
            bit_samples = bit_samples.reshape(1, -1)
        counts: Dict[str, int] = {}
        for sample in bit_samples:
            bs = "".join(map(str, sample))
            counts[bs] = counts.get(bs, 0) + 1

        results = []
        for bs, cnt in counts.items():
            x = [int(b) for b in bs]
            cost = sum(w * x[self.var_index[a]] * x[self.var_index[b]]
                       for (a, b), w in self.qubo.items())
            results.append((bs, cnt, cost))
        results.sort(key=lambda t: t[2])
        self.best_bitstring = results[0][0] if results else None
        self.bitstring_counts = results
        return exp_val

    # --------------------------- Annealing (opcional) ---------------------- #
    # ... (mantido, inalterado) ...

    # --------------------------- Visualização ----------------------------- #
    def plot_histogram(self, filename=None):
        """Plota histograma das bitstrings medidas."""
        if not self.bitstring_counts:
            print("Execute solve_qaoa() primeiro.")
            return
        import matplotlib.pyplot as plt
        labels = [bs for bs, _, _ in self.bitstring_counts]
        freqs = [cnt for _, cnt, _ in self.bitstring_counts]
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(labels)), freqs, tick_label=labels)
        plt.xticks(rotation=90)
        plt.xlabel("Bitstring")
        plt.ylabel("Frequência")
        plt.title("Histograma de estados medidos (QAOA)")
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
            print(f"Historgama {filename} criado com sucesso!!!")

# --------------------------- Teste mínimo --------------------------- #
if __name__ == "__main__":
    # Pequeno cenário sintético
    from utils import Passenger, Vehicle, RechargePoint
    from geografics import generate_random_geografic_points
    from grouper import PassengerGrouper, Assignment
    from quantum import QuboGenerator

    np.random.seed(0)
    n_passengers, n_vehicles, n_recharges = 10, 8, 1
    org = generate_random_geografic_points(n_passengers, (-1, 1), (-1, 1))
    dst = generate_random_geografic_points(n_passengers, (-1, 1), (-1, 1))
    rec = generate_random_geografic_points(n_recharges, (-1, 1), (-1, 1))

    passengers = [Passenger(i, o, d) for i, (o, d) in enumerate(zip(org, dst))]
    vehicles = [Vehicle(i, battery=30.0) for i in range(n_vehicles)]
    recharge_pts = [RechargePoint(0, rec[0])]

    grouper = PassengerGrouper(passengers, eps=0.8)
    groups = grouper.build_groups()
    assignment = Assignment(groups, vehicles)

    qgen = QuboGenerator(assignment, recharge_pts)
    qubo_dict = qgen.build_qubo()
    q_order = qgen.variables

    optimizer = QuantumRouteOptimizer(qubo_dict, q_order)
    expected_cost = optimizer.solve_qaoa(p=1, steps=40, lr=0.2, shots=300)
    print("Valor esperado mínimo:", expected_cost)
    print("Melhor bitstring:", optimizer.best_bitstring)

    optimizer.plot_histogram("figs/teste_histogram.png")