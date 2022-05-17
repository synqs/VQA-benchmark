import networkx as nx
import numpy    as np

from typing import Callable, Dict, List
from general.myTypes import Number
from numpy.typing import ArrayLike, NDArray


def get_expval_func(problem: str, n: int, penalty: Number) -> Callable[[Dict[str, int], nx.Graph], float]:
	if problem == "max_cut":
		def func(counts: Dict[str, int], G: nx.Graph) -> float:
			norm    : int = 0
			energies: Number = 0
			for state, times in counts.items():
				energy: Number = eval_maxcut(state, G)
				energies += energy * times
				norm     +=          times
			return energies / norm
	elif problem == "max_cut_full":
		def func(counts: Dict[str, int], G: nx.Graph) -> float:
			norm    : int = 0
			energies: Number = 0
			for state, times in counts.items():
				energy: Number = eval_maxcut_full(state, G)
				energies += energy * times
				norm     +=          times
			return energies / norm
	elif problem == "tsp":
		def func(counts: Dict[str, int], G: nx.Graph) -> float:
			norm    : int = 0
			energies: Number = 0
			for state, times in counts.items():
				energy: Number = eval_tsp(state, G, n, penalty)
				energies += energy * times
				norm     +=          times
			return energies / norm
	elif problem == "tsp_full":
		def func(counts: Dict[str, int], G: nx.Graph) -> float:
			norm    : int = 0
			energies: Number = 0
			for state, times in counts.items():
				energy: Number = eval_tsp_full(state, G, n, penalty)
				energies += energy * times
				norm     +=          times
			return energies / norm
	else:
		raise KeyError("Unknown problem '"+ problem +"'.")
	return func

# def expval(c, G, problem, penalty):
# 	"""Returns the expectation value of the problem Hamiltonian (= Evaluates/Averages over the objective function)."""
# 	norm     = 0
# 	energies = 0
# 	for state, times in counts.items():
# 		if problem == "max_cut":
# 			energy = eval_maxcut(state, G)
# 		elif problem == "tsp":
# 			energy = eval_tsp(state, G, n, penalty)
# 		else:
# 			raise KeyError("Unknown problem '"+ problem +"'.")
# 		energies += energy * times
# 		norm     +=          times
# 	return energies / norm

def eval_maxcut(state: str, G: nx.Graph) -> Number:
	choice: str = '0'+ state[::-1] # TODO also give '1' a try. Maybe it's better...
	cut: Number = 0
	for i, j in G.edges():
		if choice[i] != choice[j]:
			cut += G[i][j]['weight']
	return - cut # max cut = min energy

def eval_maxcut_full(state: str, G: nx.Graph) -> Number:
	choice: str = state[::-1]
	cut: Number = 0
	for i, j in G.edges():
		if choice[i] != choice[j]:
			cut += G[i][j]['weight']
	return - cut # max cut = min energy

def eval_tsp_full(state: str, G: nx.Graph, n: int, factor: Number) -> Number:
	raise NotImplementedError
	return 0

def eval_tsp(state: str, G: nx.Graph, n: int, factor: Number) -> Number:
	K: NDArray[np.float64] = nx.to_numpy_matrix(G, weight='weight', nonedge=np.infty)
	X: NDArray[np.int64] = bits2mat(state[::-1], n)
	N: range = range(n)


	s: Number
	objFun: Number = 0
	# each time exactly one node
	for p in N:
		s = 0
		for i in N:
			s += X[i,p]
		objFun += factor*(s-1)**2
					
	# each node exactly once
	for i in N:
		s = 0
		for p in N:
			s += X[i,p]
		objFun += factor*(s-1)**2

	# each edge costs its weight
	for i in N:
		for j in N:
			prev: int = list(N)[-1]
			for p in N:
				objFun += K[i, j] * X[i, prev] * X[j, p]
				prev = p

	return objFun



def bits2mat(bitstring: str, n: int) -> NDArray[np.int64]:
	line: int = n-1
	assert len(bitstring) == line * line

	table: List[str] = []
	for i in range(line):
		table.append(bitstring[i*line:(i+1)*line] +'0')
	table.append('0'*(n-1) + '1')

	return np.matrix([[int(symbol) for symbol in line] for line in table]).T

#bits2mat(a)




####
# This function transforms the state into a readable variant.
###

def readState(state: str, problem: str) -> str:
	if problem == "max_cut":
		return '0'+ state
	elif problem == "max_cut_full":
		return state
	elif problem == "tsp":
		return bits2perm(state)
	elif problem == "tsp_full":
		raise NotImplementedError
		return bits2perm(state)
	else:
		raise KeyError("Unknown problem '"+ problem +"'.")

def bits2perm(bitstring: str) -> str:
	line: int = round(np.sqrt(len(bitstring)))
	# line = n-1
	
	# print(bitstring)
	# print(len(bitstring))
	# print(line)

	permutation: str = ""
	stopATtime: List[str] = []
	for i in range(line):
		linestring = bitstring[i*line:(i+1)*line]
		# print(linestring)
		for j in range(line):
			if int(linestring[j]):
				stopATtime.append(int2str(j))
		if len(stopATtime) == 1:
			permutation += stopATtime[0]
		else:
			permutation += '('+ ''.join(stopATtime) +')'
	# assert len(permutation) == line, permutation

	# print(permutation)
	
	return permutation + str(line)

#bits2perm(a)


# Converts letter digits to higher numbers
# for numbers equals int(symbol)
# str2int('A') == 10
# str2int('B') == 11
# ...
def str2int(symbol: str) -> int:
	assert len(symbol) == 1, "invalid symbol occured"
	return int(symbol, 36)
	if symbol.isalpha():
		return int(symbol.upper(), 36)
	elif symbol.isnumeric():
		return int(symbol)
	else:
		raise TypeError()



# Converts higher numbers to letter digits
# for numbers equals str(num)
# int2str(10) == 'A'
# int2str(11) == 'B'
# ...
def int2str(num: int) -> str:
	assert num >= 0, "negative number occured"
	assert num < 36,  "invalid number occured"
	if num < 10:
		return str(num)
	else:
		return chr(55 + num)

