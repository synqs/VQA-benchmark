#!/usr/bin/env python3.9
# -*- coding:utf-8 -*-
from general.settings import all_options
from general.manager  import *


options: Dict[str, Union[str, int, float, bool]] = {
	'problem':					"max_cut_full",				# max_cut, tsp, max_cut_full, tsp_full # the latter with no classical simplification
	'size':						"tiny", 				# tiny, small, large
	'distances':				1,						# 1, 2, ... (only a few possible)
	'penalty':					100,  					# penalty for invalid tsp states
	'shots':					1024,					# shots per quantum run
	'pmax':						4,						# maximal number of evolution steps
	'qubase':					"qubit",				# qubit, qudit
	'd':						10,						# 2, 3, ...
	'qAlgorithm':				"QAOA",					# VQE_linear, VQE_all, QAOA, WS-QAOA, Grover
	'hardware':					"qasm_simulator",		# qasm_simulator, statevector_simulator, ibmq_quito
	'cAlgorithm':				"powell",				# powell, Something_with_gradients?, something_own
	'x0':						"standard",				# standard (decrease for VQE, linear_annealing for QAOA), decrease, zeros, ones, increase, large, linear_annealing
	'print_circuits':			False,					# True, False
	'print_distributions':		True,					# True, False
	'print_comparisons':		True,					# True, False
}



final_result: int = single_run(options)
# final_result: int = vary('qAlgorithm', all_options, options)
# final_result: int = vary(('qAlgorithm', 'size'), all_options, options, how_many=2)

# print(final_result)
