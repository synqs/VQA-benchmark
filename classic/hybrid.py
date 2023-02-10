import quantum.circuit
import quantum.general
import quantum.explicit
import classic.hamiltonian

import numpy as np


from general.myTypes import nx, QuantumCircuit, BaseBackend, csr, NDArray
from typing import Dict, Sequence, Callable, Tuple




def get_process(G: nx.Graph, circuit: QuantumCircuit, n: int, problem: str, hardware: str, penalty: float, shots: int) -> Callable[[Sequence[float]], float]:
	backend: BaseBackend = quantum.general.get_backend(hardware)
	get_expectation_value: Callable[[Dict[str, int], nx.Graph], float] = classic.hamiltonian.get_expval_func(problem, n, penalty)

	def process(thetas: Sequence[float]) -> float:
		counts: Dict[str, int] = quantum.circuit.run(thetas, circuit, backend, shots)
		energy: float = get_expectation_value(counts, G)
		return energy
	return process

def get_process_linalg(G: nx.Graph, quantum_objects: Tuple[Tuple[NDArray, NDArray], NDArray, NDArray], p: int, n: int, problem: str) -> Callable[[Sequence[float]], float]:
	Hmix, Hprob, init = quantum_objects
	def process(thetas: Sequence[float]) -> float:
		state: NDArray = quantum.explicit.run(thetas, Hmix, Hprob, init, p)
		energy: float   = quantum.explicit.get_expectation_value(state, Hprob)
		return energy
	return process

