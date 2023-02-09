"""
Post-processing for count dicts and optimizer results.
"""



import numpy as np

import classic.hamiltonian

from typing import Dict, Iterable #, List
from typing import Any, Optional
from general.myTypes import QuantumCircuit, Solution, Number, Result




def evaluate(optimizer_result: Any, counts: Dict[str, int], circuit: Optional[QuantumCircuit], solution: Solution, runtime: float) -> Result:
	success_rate: Optional[Number]		= None
	average_energy: Number				= 0
	quantum_entropy: Optional[Number]	= None


	average_energy = optimizer_result['fun']

	# TODO extract entropy, correlations

	if solution:
		best_inputs, best_energy = solution
		success_rate = 0
		for state, rate in counts.items():
			if state in best_inputs:
				success_rate += rate
		success_rate /= sum(counts.values())
		# average_energy /= best_energy
		average_energy = 1 - (average_energy - best_energy) / abs(best_energy)
		# average_energy -= best_energy


	

	return success_rate, average_energy, quantum_entropy, runtime



def prettify(chaos_counts: Dict[str, int], problem: str, n: int) -> Dict[str, int]:
	# return dict(sorted(chaos_counts.items(), key=lambda item: item[0]))
	names : Iterable[str] = sorted(chaos_counts)
	# values: List[int] = []
	counts: Dict[str, int] = {}
	for i in names:
		# values.append(chaos_counts[i])
		counts[classic.hamiltonian.prettyState(problem, i, n)] = chaos_counts[i]

	return counts
