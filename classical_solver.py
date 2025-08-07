"""
classical_solver.py
===================

Resolved‑route search for small EA‑VRP instances *without* quantum hardware.
The class below reuses the QUBO that `QUBOEncoder` already builds and feeds it
into a **classical** exact solver (`dimod.ExactSolver`).  Because the search is
exhaustive, it is practical only for toy examples (≈25–30 binary vars max),
but it is fully deterministic and produces the global optimum.

Usage example (sketch)
----------------------
```python
from encoder import QUBOEncoder               # seu arquivo já existente
from utils   import make_toy_instance         # helper que devolve instância

vehicles, groups, depot, recharge_points = make_toy_instance()

enc = QUBOEncoder(
    vehicles, groups, recharge_points,
    depot=depot,
    penalty_lambda=5.0,
    penalty_mu=2.0,
)

from classical_solver import ClassicalVRPSolver
solver = ClassicalVRPSolver(enc)
solver.solve()                # busca exaustiva
print(solver.best())          # melhor solução factível + rotas
```

Dependências
------------
* **dimod**  (vem com `dwave-ocean-sdk`)
* **numpy**  (já requerido no projeto)

O módulo não faz I/O externo; toda a lógica pesada é delegada ao `encoder`.
"""

from typing import Any, Dict, List, Optional, Union

import dimod
import numpy as np


class ClassicalVRPSolver:
    """Exhaustive‑search solver for *pequenas* instâncias EA‑VRP.

    Parameters
    ----------
    encoder : QUBOEncoder
        Instância já inicializada que implementa pelo menos `encode()` e
        `is_feasible(bits)`.  A função `cost(bits)` é opcional; se não existir,
        será retornado *NaN* como custo.
    sampler : dimod.Sampler, optional
        Sampler a ser usado. Padrão: :class:`dimod.ExactSolver` (exaustivo).
    """

    def __init__(self, encoder: Any, sampler: Optional[dimod.Sampler] = None):
        self.encoder = encoder

        # —‑‑‑‑‑‑ construir BQM reutilizando o QUBO já gerado ‑‑‑‑‑‑‑—
        Q, offset, _ = self.encoder.encode()
        self.bqm = dimod.BQM.from_qubo(Q, offset=offset)

        # Sampler: exaustivo por padrão
        self.sampler: dimod.Sampler = sampler or dimod.ExactSolver()
        self.response: Optional[dimod.SampleSet] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, **sample_kwargs) -> dimod.SampleSet:
        """Executa o sampler, ordena por energia e faz cache."""
        if self.response is not None:
            return self.response

        samples = self.sampler.sample(self.bqm, **sample_kwargs)

        # Garantir ordenação crescente por energia
        samples = samples.copy()  # torna mutável
        samples.record.sort(order="energy")

        self.response = samples
        return self.response

    def best(self) -> Dict[str, Any]:
        """Retorna dicionário com a melhor amostra + métricas."""
        if self.response is None:
            self.solve()

        rec = self.response.first  # já é o menor em energia
        bits = _sample_to_bitstring(rec.sample)
        return {
            "bits": bits,
            "energy": float(rec.energy),
            "feasible": self.encoder.is_feasible(bits),
            "cost": _safe_cost(self.encoder, bits),
            "routes": getattr(self.encoder, "interpret", lambda _: None)(bits),
        }

    def k_best(self, k: int = 10) -> List[Dict[str, Any]]:
        """Lista as *k* melhores amostras (factíveis ou não)."""
        if self.response is None:
            self.solve()

        out: List[Dict[str, Any]] = []
        for i in range(min(k, len(self.response))):
            rec = self.response.record[i]
            bits = _sample_to_bitstring(rec.sample)
            out.append(
                {
                    "bits": bits,
                    "energy": float(rec.energy),
                    "feasible": self.encoder.is_feasible(bits),
                    "cost": _safe_cost(self.encoder, bits),
                }
            )
        return out


# ----------------------------------------------------------------------
# Helper functions ------------------------------------------------------
# ----------------------------------------------------------------------

def _safe_cost(encoder: Any, bits: str) -> float:
    """Tenta chamar `encoder.cost(bits)`. Se não existir, devolve NaN."""
    func = getattr(encoder, "cost", None)
    if callable(func):
        return float(func(bits))
    return float("nan")


def _sample_to_bitstring(sample: Union[np.ndarray, Dict[int, int]]) -> str:
    """Converte amostra (array **ou** dict) para bitstring ordenada."""
    if isinstance(sample, np.ndarray):
        return "".join(map(str, sample.astype(int)))

    if isinstance(sample, dict):
        if not sample:
            return ""
        n = max(sample.keys()) + 1
        bits = ["0"] * n
        for idx, val in sample.items():
            bits[int(idx)] = str(int(val))
        return "".join(bits)

    raise TypeError("sample precisa ser ndarray ou dict, obtido %s" % type(sample))


# ----------------------------------------------------------------------
# Ad‑hoc smoke test -----------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    try:
        from encoder import QUBOEncoder  # type: ignore
        from utils import make_toy_instance  # helper hipotético
    except ImportError as exc:
        print("✖ Unable to import project modules:", exc, file=sys.stderr)
        sys.exit(1)

    vehs, grps, depot, rps = make_toy_instance()
    enc = QUBOEncoder(
        vehs,
        grps,
        rps,
        depot=depot,
        penalty_lambda=5.0,
        penalty_mu=2.0,
    )

    cls = ClassicalVRPSolver(enc)
    print("Melhor solução:")
    print(cls.best())
