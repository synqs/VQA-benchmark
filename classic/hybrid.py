import quantum.circuit
import quantum.general
import classic.hamiltonian


from general.myTypes import nx, QuantumCircuit, BaseBackend
from typing import Dict, Sequence, Callable




def get_process(G: nx.Graph, circuit: QuantumCircuit, n: int, problem: str, hardware: str, penalty: float) -> Callable[[Sequence[float]], float]:
	backend: BaseBackend = quantum.general.get_backend(hardware)
	get_expectation_value: Callable[[Dict[str, int], nx.Graph], float] = classic.hamiltonian.get_expval_func(problem, n, penalty)

	def process(thetas: Sequence[float]) -> float:
		counts: Dict[str, int] = quantum.circuit.run(thetas, circuit, backend)
		energy: float = get_expectation_value(counts, G)
		return energy
	return process

