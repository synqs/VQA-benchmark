from qiskit import QuantumCircuit #, QuantumRegister, ClassicalRegister

from quantum.thetas import Parameters

import networkx as nx

from typing import Tuple, List
from typing import Optional, NoReturn


def design_parameters(problem: str, algorithm: str, p: int, n: int) -> Tuple[int, int, Parameters]:
	q: int
	pars: int

	if problem == "max_cut":
		q = n-1
	elif problem == "tsp":
		q = (n-1) * (n-1)
	elif problem == "max_cut_full":
		q = n
	elif problem == "tsp_full":
		q = n * n
	else:
		raise KeyError("Unknown problem '"+ problem +"'.")
	
	if algorithm.startswith("VQE_"):
		pars =  (2*p - 1) * q
		if algorithm.endswith("_linear_rzz") or algorithm.endswith("_linear_rxx"):
			pars += (p-1) * (q-1)
		elif algorithm.endswith("_all_rzz") or algorithm.endswith("_all_rxx"):
			pars += (p-1) * int(q * (q-1) / 2)
	elif algorithm == "QAOA":
		pars = 2 * p
	elif algorithm == "Grover":
		pass
	else:
		raise KeyError("Unknown quantum algorithm '"+ algorithm +"'.")
	
	theta: Parameters = Parameters(pars)
	return q, pars, theta


def get_circuit(algorithm: str, p: int, q: int, theta: Parameters, graph: Optional[nx.Graph] = None, problem: Optional[str] = None) -> QuantumCircuit:
	circuit: QuantumCircuit = QuantumCircuit(q, q)

	if algorithm.startswith("VQE_"):
		VQE(circuit, p, q, theta, algorithm[4:])
	elif algorithm == "QAOA":
		assert problem is not None and graph is not None, "QAOA depends on the given problem"
		QAOA(circuit, p, q, theta, graph, problem, 'classic')
	elif algorithm == "WS-QAOA":
		assert problem is not None and graph is not None, "QAOA depends on the given problem"
		QAOA(circuit, p, q, theta, graph, problem, 'warm-start')
	elif algorithm == "Grover":
		Grover(circuit, p, q, theta)
	else:
		raise KeyError("Unknown quantum algorithm '"+ algorithm +"'.")
	
	assert theta.is_complete(), (algorithm, p, q, theta, problem)

	for qubit in range(q):
		circuit.measure(qubit, qubit)
	
	# circuit.draw('mpl')
	return circuit



def VQE(circuit: QuantumCircuit, p: int, q: int, theta: Parameters, style: str) -> None:
	qubits: range = range(q)
	# prefix
	for qubit in qubits:
		circuit.ry(next(theta), qubit)

	# repetition
	prev: int
	for i in range(p-1):
		# entanglement
		if style == "qiskit_linear":
			prev = 0
			for qubit in range(1,q):
				circuit.cx(prev, qubit)
				prev = qubit
		elif style == "qiskit_all":
			for qubit1 in range(q):
				for qubit2 in range(qubit1+1,q):
					circuit.cx(qubit1, qubit2)
		elif style == "linear_cz":
			prev = 0
			for qubit in range(1,q):
				circuit.cz(prev, qubit)
				prev = qubit
		elif style == "all_cz":
			for qubit1 in range(q):
				for qubit2 in range(qubit1+1,q):
					circuit.cz(qubit1, qubit2)
		elif style == "linear_rzz":
			prev = 0
			for qubit in range(1,q):
				circuit.rzz(next(theta), prev, qubit)
				prev = qubit
		elif style == "all_rzz":
			for qubit1 in range(q):
				for qubit2 in range(qubit1+1,q):
					circuit.rzz(next(theta), qubit1, qubit2)
		elif style == "linear_rxx":
			prev = 0
			for qubit in range(1,q):
				circuit.rxx(next(theta), prev, qubit)
				prev = qubit
		elif style == "all_rxx":
			for qubit1 in range(q):
				for qubit2 in range(qubit1+1,q):
					circuit.rxx(next(theta), qubit1, qubit2)
		# TODO CPHASE https://en.wikipedia.org/wiki/Quantum_logic_gate#Controlled_phase_shift
		elif style == "own":
			prev = 0
			for qubit in range(1,q):
				circuit.cz(prev, qubit)
				prev = qubit
		else:
			pass
		# rotation
		for qubit in qubits:
			circuit.rz(next(theta), qubit) # [q * (2*i + 1) + 2*qubit    ]
			circuit.ry(next(theta), qubit) # [q * (2*i + 1) + 2*qubit + 1]
	# suffix
	
	# evaluation is done in the calling function




def QAOA(circuit, p: int, q: int, theta: Parameters, graph: nx.Graph, problem: str, style: str, start = Optional[List[float]]) -> None:
	qubits: range = range(q)
	# prefix
	if style == "classic":
		for qubit in qubits:
			circuit.h(qubit)
	elif style == "warm-start":
		for qubit in qubits:
			circuit.rz(start[qubit], qubit)
	else:
		raise KeyError("Unknown QAOA style '"+ style +"'.")
	
	# repetition
	fixed_node: int = q
	for i in range(p):
		# goal hamiltonian
		gamma_i = next(theta)
		if problem == "max_cut":
			for s, e, d in graph.edges(data='weight'):
				# print(s, e, d, fixed_node)
				if s == fixed_node:
					circuit.rz( gamma_i * d, e)
				elif e == fixed_node:
					circuit.rz( gamma_i * d, s)
				else:
					circuit.rzz(gamma_i * d, s, e)
				
			# for qubit1 in range(q):
			# 	for qubit2 in range(qubit1+1,q):
			# 		circuit.rzz(gamma_i * graph[qubit1][qubit2]["weight"], qubit1, qubit2)
			# for qubit in range(q):
			# 	circuit.rz(gamma_i * graph[qubit][fixed_node]["weight"], qubit)
			# 	prev = qubit
		elif problem == "max_cut_full":
			for s, e, d in graph.edges(data='weight'):
				circuit.rzz(gamma_i * d, s, e)
		elif problem == "tsp":
			pass
			# K: NDArray[np.float64] = nx.to_numpy_matrix(G, weight='weight', nonedge=0)
			# X: NDArray[np.int64] = bits2mat(state, n)
			# N: range = range(n)


			# s: Number
			# objFun: Number = 0
			# # each time exactly one node
			# for p in N:
			# 	s = 0
			# 	for i in N:
			# 		s += X[i,p]
			# 	objFun += factor*(s-1)**2
							
			# # each node exactly once
			# for i in N:
			# 	s = 0
			# 	for p in N:
			# 		s += X[i,p]
			# 	objFun += factor*(s-1)**2

			# # each edge costs its weight
			# for i in N:
			# 	for j in N:
			# 		prev: int = list(N)[-1]
			# 		for p in N:
			# 			objFun += K[i, j] * X[i, prev] * X[j, p]
			# 			prev = p
		elif problem == "tsp_full":
			pass
			# K: NDArray[np.float64] = nx.to_numpy_matrix(G, weight='weight', nonedge=0)
			# X: NDArray[np.int64] = bits2mat(state, n)
			# N: range = range(n)


			# s: Number
			# objFun: Number = 0
			# # each time exactly one node
			# for p in N:
			# 	s = 0
			# 	for i in N:
			# 		s += X[i,p]
			# 	objFun += factor*(s-1)**2
							
			# # each node exactly once
			# for i in N:
			# 	s = 0
			# 	for p in N:
			# 		s += X[i,p]
			# 	objFun += factor*(s-1)**2

			# # each edge costs its weight
			# for i in N:
			# 	for j in N:
			# 		prev: int = list(N)[-1]
			# 		for p in N:
			# 			objFun += K[i, j] * X[i, prev] * X[j, p]
			# 			prev = p
		else:
			raise KeyError("Unknown problem '"+ problem +"'.")
		# mixing hamiltonian
		if style == "classic":
			beta_i = next(theta)
			for qubit in qubits:
				circuit.rx(beta_i, qubit)
		if style == "warm-start":
			pass
		else:
			pass
	# suffix
	
	# evaluation is done in the calling function


def Grover(circuit: QuantumCircuit, p: int, q: int, theta: Parameters) -> NoReturn:
	raise NotImplementedError("Implement Grover algorithm")