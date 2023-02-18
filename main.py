#!/usr/bin/env python3.8
# -*- coding:utf-8 -*-
from general.settings import all_options
from general.manager  import *


options: Dict[str, Union[str, int, float, bool]] = {
	'problem':					"MCP",					# MCP, TSP, MCP_full, TSP_full # the latter with no classical simplification
	'size':						"small", 				# tiny, small, medium, large
	'distances':				1,						# 1, 2 (only possible for small)
	'penalty':					100,  					# penalty for invalid tsp states
	'shots':					1024,					# shots per quantum run
	'pmax':						4,						# maximal number of evolution steps
	'qubase':					"qubit",				# qubit, qudit (not implemented)
	# 'd':						10,						# 2, 3, ...
	'platform':					"qiskit",				# qiskit (circuit), linalg (matrices), (not implemented: qutip, sympy)
	'qAlgorithm':				"VQE",  				# QAOA, cQAOA, VQE_qiskit_linear, VQE_qiskit_all, VQE_linear_cz, VQE_all_cz
	'hardware':					"qasm_simulator",		# qasm_simulator, statevector_simulator, ibmq_quito
	'cAlgorithm':				"powell",				# powell, Something_with_gradients?, something_own
	'x0':						"standard",				# standard (decrease for VQE, linear_annealing for QAOA), decrease, zeros, ones, increase, large, linear_annealing, random
	'print_circuits':			False,					# True, False
	'print_distributions':		True,					# True, False
	'print_comparisons':		True,					# True, False
}

np.random.seed(42)


# final_result: int = single_run(options)
final_result: int = vary('problem', all_options, options)
# final_result: int = vary(('qAlgorithm', 'size'), all_options, options, how_many=2)

# print(final_result)
