import numpy as np
from scipy.optimize import minimize
import time as clock

from typing import Optional, Tuple, Callable, Sequence
from numpy.typing import ArrayLike, NDArray
from general.myTypes import OptimizeResult



def get_start_value(pars: int, style: str, qAlgorithm: Optional[str] = None) -> NDArray[np.float64]:
	if style == "standard":
		assert qAlgorithm, "Default behaviour depends on the quantum algorithm, no algorithm given."
		if qAlgorithm.startswith("VQE"):
			return get_start_value(pars, "decrease")
		elif qAlgorithm.startswith("QAOA") or qAlgorithm.startswith("cQAOA"):
			return get_start_value(pars, "linear_annealing")
		else:
			raise KeyError("Unknown quantum algorithm '"+ qAlgorithm +"'.")
	elif style == "decrease": # small, decreasing start parameters > 0
		return 0.1 / (np.arange(pars) + 1)
	elif style == "increase": # small, increasing start parameters > 0
		return 0.1 / (np.arange(pars, 0, -1))
	elif style == "zeros":
		return np.zeros(pars)
	elif style == "ones":
		return np.ones(pars)
	elif style == "large": # some explanation
		return np.pi / (np.arange(pars) + 1)
	elif style == "linear_annealing": # for QAOA (assuming values of 10~1000 in the problem)
		mixer  : NDArray[np.float64] = np.arange(1/2,  0,  - 1 / pars) * np.pi / 1000 # betas
		# problem: NDArray[np.float64] = np.arange(0,  1/2,  + 1 / pars) * np.pi
		problem: NDArray[np.float64] = np.array(list(reversed(mixer))) / 100 # gammas
		# divided by 100 as the rotation factors = edge weights and penalties tend to be of order 10-1000
		return np.array([problem, mixer]).flatten('F')
	elif style == "random":
		return np.random.random_sample(pars) * np.pi / 100
	elif style == "random_large":
		return np.random.random_sample(pars) * np.pi / 2
	else:
		raise KeyError("Start value style '"+ style +"' not known.")

def run(func: Callable, x0: ArrayLike, method: str) -> Tuple[OptimizeResult, float]:
	start: float = clock.time()
	minimum: OptimizeResult = minimize(func, x0, method=method
							, options={'xtol': 1e-1, 'disp': False}) # TODO is xtol ok?
	stop: float = clock.time()
	duration: float = stop - start
	print(f"This run took {duration:4.0f} s and resulted in a minimum of {minimum.fun:6.1f}.")
	return minimum, stop - start


