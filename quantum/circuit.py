from qiskit import transpile#, BasicAer
import matplotlib.pyplot as plt
import numpy as np

# import hamiltonian

from general.myTypes import QuantumCircuit, BaseBackend, nx, Solution
from typing import Dict, Sequence
from typing import Union, Optional



def run(thetas: Sequence[float], circuit: QuantumCircuit, backend: BaseBackend) -> Dict[str, int]:
	circuit_instance: QuantumCircuit = circuit.bind_parameters(thetas)
	
	job = backend.run(transpile(circuit_instance, backend))
	counts: Dict[str, int] = job.result().get_counts()
	
	# return dict(sorted(counts.items(), key=lambda item: item[0]))

	# names  = sorted(chaos_counts)
	# # values = []
	# counts = {}
	# for i in names:
	# 	# values.append(chaos_counts[i])
	# 	counts[hamiltonian.readState(i)] = chaos_counts[i]
	
	return counts



def visualize(c: QuantumCircuit, thetas: Sequence[float], file: str) -> QuantumCircuit:
	the_circuit: QuantumCircuit = c.bind_parameters([round(display, 3) for display in thetas])
	# np.save(file.replace("img", "data"+ os.sep +"img").replace(".png", ""), thetas) # is done after the function in benchmark.py
	the_circuit.draw('mpl', filename=file)
	plt.close('all')
	return the_circuit
