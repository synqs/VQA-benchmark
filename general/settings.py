
all_options: Dict[str, Iterable] = {
	'problem':		["max_cut", "tsp"],
	'size':			["small", #"large",
								"tiny"],
	'distances':	range(1, 3),
	'penalty':		[10, 100, 1000],
	# 'pmax':		4,
	'qubase':		["qubit", "qudit"],
	'd':			range(2, 12),
	# 'qAlgorithm':	["QAOA", "VQE_qiskit_linear", "VQE_qiskit_all", "VQE_linear_cz", "VQE_all_cz", "VQE_linear_rzz", "VQE_all_rzz", "VQE_linear_rxx", "VQE_all_rxx"], # "VQE_own", "WS-QAOA", "Grover"],
	'qAlgorithm':	["QAOA"],#, "VQE_linear_cz", "VQE_all_cz"],
	'hardware':		["qasm_simulator", "statevector_simulator", "ibmq_quito"],
	'cAlgorithm':	["powell", "Something_with_gradients?", "something_own"],
	'x0':			["decrease", "zeros", "ones", "increase", "large", "linear_annealing"],
	'print_circuit_images':	[True, False],
	'print_distributions':	[True, False],
	'print_comparisons':	[True, False],
}

