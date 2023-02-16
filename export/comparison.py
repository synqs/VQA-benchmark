import matplotlib.pyplot as plt
import numpy as np
import os

from typing import Dict, List, Tuple
from typing import Union, Optional
from general.myTypes import Solution, Number, Result, OptimizeResult





# Update: the following should be mostly replaced by the replot.py TODO update bash command
# To see your changes while debugging you can open an image (./storage/img/compare/QAOA-MCP-small-distances-4.png) and run the following in bash:
# while sleep 1; do if [ "./storage/img/compare/QAOA-MCP-small-distances-4.png" -ot export/comparison.py ]; then echo -n "and again "; if python3.9 -c 'import numpy as np; from export.comparison import plot; reports, maxP, fileName = np.load("./storage/data/img/compare/QAOA-MCP-small-distances-4_all.npy", allow_pickle=True); plot(reports, maxP, fileName)'; then echo 'a new plot is done'; else echo 'this failed'; sleep 14; fi; fi; done
def plot(reports: List[Tuple[Tuple[List[Result], Dict[str, int], Solution, OptimizeResult], Union[Tuple[()], Tuple[str], Tuple[str, str], Tuple[str, str, str]], Union[Tuple[()], Tuple[str], Tuple[str, str], Tuple[str, str, str]]]], maxP: int, file: str) -> None:
	# plt.style.use('Solarize_Light2')

	SMALL_SIZE  = 28
	MEDIUM_SIZE = 32
	BIGGER_SIZE = 36
	LARGE_SIZE  = 40

	plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes (= subplot) title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	# plt.rc('figure', labelsize=MEDIUM_SIZE)  # fontsize of the figure labels
	plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

	x_labels: range = range(1, maxP+1)
	fig, (ax1, ax2, ax4) = plt.subplots(3, 1, sharex=True,figsize=(16,24))
	
	plt.subplots_adjust(left=0.275,
						right=0.83, 
						bottom=0.075, 
						top=0.92, 
						hspace=0.2)
	
	# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True,figsize=(12,8))
	
	# plt.subplots_adjust(left=0.12,
	# 					bottom=0.12, 
	# 					right=0.97, 
	# 					top=0.85, 
	# 					wspace=0.25, 
	# 					hspace=0.3)
	

	properties: List[str] = file.split(os.sep)[-1].replace('.png', '').split('-')[:-1]
	varied: Union[Tuple[()], Tuple[str], Tuple[str, str], Tuple[str, str, str]] = reports[0][2]
	varied_properties: List[str] = []
	v_properties: str
	for varied_property in varied:
		try:
			properties.remove(varied_property)
		except ValueError:
			# qAlgoVariant: bool = False
			if varied_property == 'qAlgorithm':
				try:
					properties.remove('qAlgorithms')
					# qAlgoVariant = True
				except ValueError:
					try:
						properties.remove('VQEs')
						# qAlgoVariant = True
					except ValueError:
						raise Exception("qAlgorithm missing in file name.")
			else:
				print(varied_property +" does not appear in "+ str(properties))
				file = '-'.join(file.split('-')[:-1]) +'-'+ varied_property +'-'+ file.split('-')[-1]
		known_properties = {'cAlgorithm':	'classical optimizers',
							'distances':	'graphs',
							'hardware':		'hardware',
							'penalty':		'penalties',
							'platform':		'algorithms and platforms', # TODO weil bisher nur der neue algorithmus auf der neuen Plattform geht.
							'problem':		'problems',
							'qAlgorithm':	'quantum algorithms',
							'shots':		'shots per simulation',
							'size':			'sizes',
							'x0':			'initial parameters $\\theta_j$',
						}
		plural: str
		try:
			plural = known_properties[varied_property]
		except KeyError:
			plural = varied_property + 's'
		varied_properties.append(plural)
	vlen = len(varied_properties)
	if vlen > 1:
		varied_properties[-2] += ' and '+ varied_properties[-1]
		v_properties = ', '.join(varied_properties[:-1])
	elif vlen:
		v_properties = varied_properties[0]
	

	fig.suptitle((v_properties.capitalize() +' in ' if vlen else '') + ' - '.join(properties))
	fig.supxlabel('Number of quantum circuit layers $p$', fontsize=MEDIUM_SIZE)
	ax1.set_title('Success rates',             pad=12)
	ax2.set_title('Normalized average energy', pad=12)
	# ax3.set_title('More parameters',         pad=12)
	ax4.set_title('Runtime',                   pad=12)

	ax1.set_ylabel(ylabel='Ratio',               labelpad=16)
	ax2.set_ylabel(ylabel='Relative to optimum', labelpad=16)
	# ax3.set_ylabel(ylabel='Very fancy',        labelpad=16)
	ax4.set_ylabel(ylabel='in seconds',          labelpad=16)

	ax1.set_xticks(x_labels)
	ax2.set_xticks(x_labels)
	# ax3.set_xticks(x_labels)
	ax4.set_xticks(x_labels)

	for report, options, report_varied in reports:
		results, counts, sol, res = report
		run = ', '.join(options)
		assert report_varied == varied

		success_rates    : List[Optional[Number]] = []
		average_energies : List[Optional[Number]] = []
		quantum_entropies: List[Optional[Number]] = []
		runtimes         : List[Optional[Number]] = []
		for result in results:
			success_rate, average_energy, quantum_entropy, runtime = result
			success_rates    .append(success_rate)
			average_energies .append(average_energy)
			quantum_entropies.append(quantum_entropy)
			runtimes         .append(runtime)
		
		ax1.plot(x_labels, success_rates    , label=run)
		ax2.plot(x_labels, average_energies , label=run)
		# ax3.plot(x_labels, quantum_entropies, label=run)
		ax4.semilogy(x_labels, runtimes     , label=run)
	if vlen:
		ax4.legend()
	fig.savefig(fname=file)
	plt.close(fig)


def save(reports: List[Tuple[Tuple[List[Result], Dict[str, int], Solution, OptimizeResult], Union[Tuple[()], Tuple[str], Tuple[str, str], Tuple[str, str, str]], Union[Tuple[()], Tuple[str], Tuple[str, str], Tuple[str, str, str]]]], maxP: int, file: str) -> None:

	properties: List[str] = file.split(os.sep)[-1].replace('.png', '').split('-')[:-1]
	varied: Union[Tuple[()], Tuple[str], Tuple[str, str], Tuple[str, str, str]] = reports[0][2]
	for varied_property in varied:
		if varied_property not in properties and varied_property != 'qAlgorithm':
			file = '-'.join(file.split('-')[:-1]) +'-'+ varied_property +'-'+ file.split('-')[-1]

	all_success_rates    : List[List[Optional[Number]]] = []
	all_average_energies : List[List[Optional[Number]]] = []
	all_quantum_entropies: List[List[Optional[Number]]] = []
	all_runtimes         : List[List[Optional[Number]]] = []
	for report, options, varied in reports:
		results, counts, sol, res = report

		success_rates    : List[Optional[Number]] = []
		average_energies : List[Optional[Number]] = []
		quantum_entropies: List[Optional[Number]] = []
		runtimes         : List[Optional[Number]] = []
		for result in results:
			success_rate, average_energy, quantum_entropy, runtime = result
			success_rates    .append(success_rate)
			average_energies .append(average_energy)
			quantum_entropies.append(quantum_entropy)
			runtimes         .append(runtime)
	

		all_success_rates    .append(success_rates)
		all_average_energies .append(average_energies)
		all_quantum_entropies.append(quantum_entropies)
		all_runtimes         .append(runtimes)

	all_data: np.typing.NDArray[np.float64] = np.array([all_success_rates, all_average_energies, all_quantum_entropies, all_runtimes], dtype=np.float64)
	np.save(file.replace("img", "data"+ os.sep +"img").replace(".png", ""), all_data)
	np.save(file.replace("img", "data"+ os.sep +"img").replace(".png", "_all"), np.array([reports, maxP, file], dtype=object))
	# return all_data
