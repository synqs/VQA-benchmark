from graphs import graphs
import quantum.layout
import classic.hamiltonian
import classic.hybrid
import classic.optimization
import quantum.circuit
import quantum.general

import export.storage
import export.reporting
import export.distribution
import export.comparison


import itertools
import numpy as np
import matplotlib.pyplot as plt

from general.myTypes import *


def run_optimizer(options: Dict[str, Any]) -> Tuple[List[Result], Dict[str, int], Solution, OptimizeResult]:
	results: List[Result] = []

	# get some graph
	G: nx.Graph
	nodes: int # number of nodes
	solution: Solution # optimal solution
	G, nodes, solution = graphs.get(options['size'], options['problem'], options['distances'])

	pars: int
	for pMinusOne in range(options['pmax']):
		options['p'] = pMinusOne + 1
		# options['p'] = options['pmax'] # for running a single value only

		qAlgorithm: str = options['qAlgorithm']
		if qAlgorithm == 'VQE':
			qAlgorithm = 'VQE_linear_rzz'

		if qAlgorithm == 'cQAOA' and options['platform'] == 'qiskit': # fix diagram description
			qAlgorithm = 'VQE_linear_rzz'
			options['qAlgorithm'] = 'VQE'
		if qAlgorithm == 'VQE' and options['platform'] == 'linalg':   # restore correct value if necessary
			qAlgorithm = 'cQAOA'
			options['qAlgorithm'] = 'cQAOA'

		# calculate the number of qubits, and generate generic parameters
		q: Optional[int]
		theta: Optional[Parameters]
		q, pars, theta = quantum.layout.design_parameters(options['problem'], qAlgorithm, options['platform'], options['p'], nodes)
		
		# Sample reasonable starting values
		x0: NDArray = classic.optimization.get_start_value(pars, options['x0'], qAlgorithm)

		process: Callable[[Sequence[float]], float]
		res: OptimizeResult
		runtime: float

		raw_counts: Dict[str, int]
		counts    : Dict[str, int]
		if options['platform'] == 'qiskit':
			assert   q   is not None
			assert theta is not None
			assert isinstance(theta, Parameters_qiskit)

			# create the circuit
			myCircuit: QuantumCircuit = quantum.layout.get_circuit(qAlgorithm, options['platform'], options['p'], q, theta, G, options['problem'])
			if options['print_circuits']:
				myCircuit.draw('mpl', filename=export.storage.files['circuit_thetas'].format(**options))
				plt.close()
			
			# create a process callable for the classical optimizer
			process = classic.hybrid.get_process(G, myCircuit, nodes, options['problem'], options['hardware'], options['penalty'], options['shots'])
			
			#######
			# Run #
			#######
			res, runtime = classic.optimization.run(process, x0, options['cAlgorithm'])

			if options['print_circuits']:
				final_circuit: QuantumCircuit = quantum.circuit.visualize(myCircuit, res['x'], export.storage.files['circuit_values'].format(**options))
			np.save(export.storage.files['optimal_thetas'].format(**options), res['x'])
			
			raw_counts = quantum.circuit.run(res['x'], myCircuit, quantum.general.get_backend(options['hardware']), options['shots'])
			counts = export.reporting.prettify(raw_counts, options['problem'], nodes)
		elif options['platform'] == 'linalg':
			# calculate the number of qubits, and generate generic parameters
			# nodes = min(nodes, 6)
			# q: int
			full: bool = options['problem'].endswith('_full')
			n = nodes - int(not full)


			# create the necessary operators to repeat
			basis = list(itertools.permutations(range(n)))
			Λmix:  NDarray
			Vmix:  NDarray
			d_tm: float # diagonalization time
			Λmix, Vmix, d_tm               = quantum.explicit.tsp_mixer_unitary(n, basis)
			Hprob: NDarray                 = quantum.explicit.tsp_problem_hamiltonian(nodes, basis, G, full)
			Hmix:  Tuple[NDarray, NDarray] = Λmix, Vmix

			# Uprob: Callable[[float], csr]    = quantum.explicit.tsp_problem_unitary(Hprob)
			
			# create an initial state
			initial_state: NDArray = quantum.explicit.tsp_initial(n, basis)



			# create a process callable for the classical optimizer
			process = classic.hybrid.get_process_linalg(G, (Hmix, Hprob, initial_state), options['p'])

			
			#######
			# Run #
			#######
			res, runtime = classic.optimization.run(process, x0, 'CG') # options['cAlgorithm'])

			np.save(export.storage.files['optimal_thetas'].format(**options), res['x'])
			
			runtime = runtime * options['shots'] + d_tm # make a fairer comparison value, d_tm = diagonalization time
			best_state: NDArray        = quantum.explicit.run(res['x'], Hmix, Hprob, initial_state, options['p'])
			raw_counts: Dict[str, int] = dict(zip([classic.hamiltonian.mat2bits(classic.hamiltonian.tup2mat(tup)) for tup in basis], np.real(np.conj(best_state)*best_state)))
			counts = export.reporting.prettify(raw_counts, options['problem'], nodes)
		else:
			raise KeyError("Unknown platform '"+ options['platform'] +"'.")
		
		if options['print_distributions']:
			export.distribution.plot(counts, export.storage.files['distributions'].format(**options), solution)
		export    .distribution.save(counts, export.storage.files['distributions'].format(**options), solution)
		results.append(export.reporting.evaluate(res, counts, myCircuit if options['platform'] == 'qiskit' else None, solution, runtime))
		# break # for running a single value only
	return results, counts, solution, res # TODO: Only results and solution really make sense here.


def vary(this: Union[None, Tuple[()], str, Tuple[str], Tuple[str, str], Tuple[str, str, str]], from_options: AllOptions, other_options: Options, how_many: Optional[int] = None) -> int:
	one: str
	two: str
	three: str
	options: Options
	option_texts: Options = other_options.copy()
	report: Tuple[Tuple[List[Result], Dict[str, int], Solution, OptimizeResult], Union[Tuple[()], Tuple[str], Tuple[str, str], Tuple[str, str, str]], Union[Tuple[()], Tuple[str], Tuple[str, str], Tuple[str, str, str]]]
	reports: List[Tuple[Tuple[List[Result], Dict[str, int], Solution, OptimizeResult], Union[Tuple[()], Tuple[str], Tuple[str, str], Tuple[str, str, str]], Union[Tuple[()], Tuple[str], Tuple[str, str], Tuple[str, str, str]]]] = []
	if this is None or this == ():
		report = run_optimizer(other_options), (), ()
		reports.append(report)
		if other_options['print_comparisons']:
			export.comparison.plot(reports, other_options['pmax'], export.storage.files['comparisons'].format(**option_texts))#, export.storage.files['runtimes'].format(**option_texts))
		export    .comparison.save(reports, other_options['pmax'], export.storage.files['comparisons'].format(**option_texts))#, export.storage.files['runtimes'].format(**option_texts))
	elif how_many is None or how_many == 1:
		if how_many:
			# assert isinstance(this, tuple)
			this = this[0] # type: ignore
		assert isinstance(this, str), "vary was called with a ill fitting how_many-Parameter for the varied option(s)"
		for this_option in from_options[this]:
			options = other_options.copy()
			options[this] = this_option
			option_texts[this] = this
			report = run_optimizer(options), (str(this_option), ), (this, )
			reports.append(report)
			# save and plot after each run to prevent loss
			if other_options['print_comparisons']:
				export.comparison.plot(reports, other_options['pmax'], export.storage.files['comparisons'].format(**option_texts))#, export.storage.files['runtimes'].format(**option_texts))
			export    .comparison.save(reports, other_options['pmax'], export.storage.files['comparisons'].format(**option_texts))#, export.storage.files['runtimes'].format(**option_texts))
	elif how_many == 2:
		one, two = this # type: ignore
		for one_option in from_options[one]:
			assert isinstance(one, str), "vary was called with a ill fitting how_many-Parameter for the varied option(s)"
			for two_option in from_options[two]:
				assert isinstance(two, str), "vary was called with a ill fitting how_many-Parameter for the varied option(s)"
				options = other_options.copy()
				options[one] = one_option
				options[two] = two_option
				option_texts[one] = one
				option_texts[two] = two
				report = run_optimizer(options), (str(one_option), str(two_option)), (one, two)
				reports.append(report)
				# save and plot after each run to prevent loss
				if other_options['print_comparisons']:
					export.comparison.plot(reports, other_options['pmax'], export.storage.files['comparisons'].format(**option_texts))#, export.storage.files['runtimes'].format(**option_texts))
				export    .comparison.save(reports, other_options['pmax'], export.storage.files['comparisons'].format(**option_texts))#, export.storage.files['runtimes'].format(**option_texts))
	elif how_many == 3:
		one, two, three = this # type: ignore
		for one_option in from_options[one]:
			assert isinstance(one, str), "vary was called with a ill fitting how_many-Parameter for the varied option(s)"
			for two_option in from_options[two]:
				assert isinstance(two, str), "vary was called with a ill fitting how_many-Parameter for the varied option(s)"
				for three_option in from_options[three]:
					assert isinstance(three, str), "vary was called with a ill fitting how_many-Parameter for the varied option(s)"
					options = other_options.copy()
					options[one] = one_option
					options[two] = two_option
					options[three] = three_option
					option_texts[one] = one
					option_texts[two] = two
					option_texts[three] = three
					report = run_optimizer(options), (str(one_option), str(two_option), str(three_option)), (one, two, three)
					reports.append(report)
					# save and plot after each run to prevent loss
					if other_options['print_comparisons']:
						export.comparison.plot(reports, other_options['pmax'], export.storage.files['comparisons'].format(**option_texts))#, export.storage.files['runtimes'].format(**option_texts))
					export    .comparison.save(reports, other_options['pmax'], export.storage.files['comparisons'].format(**option_texts))#, export.storage.files['runtimes'].format(**option_texts))
	else:
		raise NotImplementedError("Too many options varied at once.")

	

	# if other_options['print_comparisons']:
	# 	export.comparison.plot(reports, other_options['pmax'], export.storage.files['comparisons'].format(**option_texts))#, export.storage.files['runtimes'].format(**option_texts))
	# export    .comparison.save(reports, other_options['pmax'], export.storage.files['comparisons'].format(**option_texts))#, export.storage.files['runtimes'].format(**option_texts))
	
	return 0

def single_run(options: Options) -> int:
	return vary(None, {}, options)