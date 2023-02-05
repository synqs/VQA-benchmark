import matplotlib.pyplot as plt
import numpy as np
import os

from typing import Dict, List
from general.myTypes import Solution



# Update: the following should be mostly replaced by the replot.py TODO update bash command
# To see your changes while debugging you can open an image (./storage/img/distribution/QAOA-MCP-small-2.png) and run the following in bash:
# while sleep 1; do if [ "./storage/img/distribution/QAOA-MCP-small-2.png" -ot export/distribution.py ]; then echo -n "and again "; if python3.9 -c 'import numpy as np; from export.distribution import plot; states, rates = np.load("./storage/data/img/distribution/QAOA-MCP-small-2.npy", allow_pickle=True); topStat, minE = np.load("./storage/data/img/distribution/QAOA-MCP-small-2_sol.npy", allow_pickle=True); counts = {k:v for k,v in zip(states, rates)}; plot(counts, "storage/img/distribution/QAOA-MCP-small-2.png", (topStat, minE))'; then echo 'a new plot is done'; else echo 'this failed'; sleep 14; fi; fi; done
def plot(c: Dict[str, int], file: str, solution: Solution) -> None:
	optimal_states: List[str] = []
	if solution:
		optimal_states, _ = solution
	colors: List[str] = ['tab:red' if state in optimal_states else 'tab:blue' for state in c]
	# print(colors)
	# print(c)

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

	fig, ax = plt.subplots(figsize=(16, 9)) # Create a figure containing a single axes.
	plt.subplots_adjust(left=0.10,
						right=0.97, 
						bottom=0.25, 
						top=0.99) # 0.90 for title
	
	ax.bar(range(len(c)), c.values(), tick_label=list(c.keys()), color=colors)
	plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')


	# fig.suptitle('QAOA with 2 layers for MCP on small graph')
	# fig.suptitle('QAOA with 2 layers for MCP on small graph')
	ax.set_xlabel('Measured outcome',  labelpad=10)
	ax.set_ylabel('Measurement count', labelpad=20)

	fig.savefig(fname=file)

	# plot_histogram(counts, color=colors, filename=storage.files['histograms'].format(**options)) # Da ist irgendwie nichts passiert
	plt.close(fig)

def save(c: Dict[str, int], file: str, solution: Solution) -> None:

	data = np.array([list(c.keys()), list(c.values())], dtype=object)
	np.save(file.replace("img", "data"+ os.sep +"img").replace(".png", ""), data)
	np.save(file.replace("img", "data"+ os.sep +"img").replace(".png", "_sol"), np.array(solution, dtype=object))
