import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, List
from general.myTypes import Solution



def plot(c: Dict[str, int], file: str, solution: Solution) -> None:
	optimal_states: List[str] = []
	if solution:
		optimal_states, _ = solution
	colors: List[str] = ['tab:red' if state in optimal_states else 'tab:blue' for state in c]
	# print(colors)
	# print(c)

	fig, ax = plt.subplots()  # Create a figure containing a single axes.
	# ax.plot([1, 2, 3, 4], [1, 4, 2, 3]);  # Plot some data on the axes.
	ax.bar(range(len(c)), c.values(), tick_label=list(c.keys()), color=colors)
	fig.savefig(fname=file)
	plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

	# plot_histogram(counts, color=colors, filename=storage.files['histograms'].format(**options)) # Da ist irgendwie nichts passiert
	plt.close(fig)

def save(c: Dict[str, int], file: str, solution: Solution) -> None:

	data = np.array([list(c.keys()), list(c.values())], dtype=object)
	np.save(file.replace("img", "data"+ os.sep +"img").replace(".png", ""), data)
	np.save(file.replace("img", "data"+ os.sep +"img").replace(".png", "_sol"), np.array(solution, dtype=object))
