import networkx as nx
import numpy    as np

from typing import Callable, Dict, List, Tuple
from general.myTypes import Number, Choice
from numpy.typing import ArrayLike, NDArray
from pyparsing import nested_expr

def get_expval_func(problem: str, n: int, penalty: Number) -> Callable[[Dict[str, int], nx.Graph], float]:
	decode   : Callable[[str], Choice] = readState(problem, n)
	evaluator: Callable[[Choice, nx.Graph], Number]
	if problem.startswith("MCP"):
		evaluator = eval_maxcut
	elif problem.startswith("TSP"):
		evaluator = evaluator_tsp(n, penalty)
	else:
		raise KeyError("Unknown problem '"+ problem +"'.")

	def func(counts: Dict[str, int], G: nx.Graph) -> float:
		norm    : int = 0
		energies: Number = 0
		for state, times in counts.items():
			energy: Number = evaluator(decode(state), G)
			energies += energy * times
			norm     +=          times
		return energies / norm
	return func

# def expval(c, G, problem, penalty):
# 	"""Returns the expectation value of the problem Hamiltonian (= Evaluates/Averages over the objective function)."""
# 	norm     = 0
# 	energies = 0
# 	for state, times in counts.items():
# 		if problem == "MCP":
# 			energy = eval_maxcut(state, G)
# 		elif problem == "TSP":
# 			energy = eval_tsp(state, G, n, penalty)
# 		else:
# 			raise KeyError("Unknown problem '"+ problem +"'.")
# 		energies += energy * times
# 		norm     +=          times
# 	return energies / norm

def eval_maxcut(choice: Choice, G: nx.Graph) -> Number:
	assert isinstance(choice, str)
	cut: Number = 0
	for i, j in G.edges():
		if choice[i] != choice[j]:
			cut += G[i][j]['weight']
	return - cut # max cut = min energy

def evaluator_tsp(n: int, factor: Number) -> Callable[[Choice, nx.Graph], Number]:
	def eval_tsp(X: Choice, G: nx.Graph) -> Number:
		assert isinstance(X, np.matrix)
		K: NDArray[np.float64] = nx.to_numpy_matrix(G, weight='weight') #, nonedge=np.infty)
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
	return eval_tsp


def bits2mat_callable(full: bool, n: int) -> Callable[[str], np.matrix]:
	def bits2mat(state: str) -> np.matrix:
		line: int = n if full else n-1
		assert len(state) == line * line

		bitstring = state[::-1]
		table: List[str] = []
		for i in range(line):
			table.append(bitstring[i*line:(i+1)*line] +('' if full else '0'))
		if not full:
			table.append('0'*(n-1) + '1')

		return np.matrix([[int(symbol) for symbol in line] for line in table])
	return bits2mat





def readState(problem: str, n: int) -> Callable[[str], Choice]:
	full = problem.endswith("_full")
	if problem.startswith("MCP"):
		return lambda state: state[::-1] + ('' if full else '0') # TODO also give '1' a try. Maybe it's better...
	elif problem.startswith("TSP"):
		return bits2mat_callable(full, n)
	raise KeyError("Unknown problem '"+ problem +"'.")

"""This function transforms the state into a readable variant."""
def prettyState(problem: str, bitstring: str, n: int) -> str:
	choice: Choice = readState(problem, n)(bitstring)
	if problem.startswith("MCP"):
		assert isinstance(choice, str)
		return choice
	elif problem.startswith("TSP"):
		assert isinstance(choice, np.matrix)
		return mat2perm(choice)
	raise KeyError("Unknown problem '"+ problem +"'.")

def mat2perm(binary_table: np.matrix) -> str:
	lines: Tuple[int, ...] = binary_table.shape
	assert lines[0] == lines[1], "Rectangular binary table"
	line: int = lines[0]
	# line = n-1
	
	permutation: str = ""
	for column in np.asarray(binary_table.T):
		stop_at_time: List[str] = []
		for k in range(line):
			if column[k]:
				stop_at_time.append(int2str(k))
		if len(stop_at_time) == 1:
			permutation += stop_at_time[0]
		else:
			permutation += '('+ ''.join(stop_at_time) +')'
	# assert len(permutation) == line, permutation

	# print(permutation)
	# next = int2str(line)

	return permutation# , next

def perm2mat(permutation: str) -> np.matrix:
	# lines: Tuple[int, int] = binary_table.shape
	# line = n-1
	
	# print(bitstring)
	# print(len(bitstring))
	# print(line)

	columns: List[List[int]] = []
	currcol: List[int]
	cursor : int             = 0
	inbrack: bool            = False
	for letter in permutation:
		if letter == "(":
			inbrack = True
			currcol = []
		elif letter == ")":
			columns.append(currcol)
			inbrack = False
		else:
			row = str2int(letter)
			if inbrack:
				currcol.append(row)
			else:
				columns.append([row])
	
	line: int = len(columns)
	emptyCol: NDArray = np.zeros(line, dtype=int)

	npCol:       NDArray
	npCols: List[NDArray] = []
	for column in columns:
		npCol = emptyCol.copy()
		for row in column:
			npCol[row] = 1
		npCols.append(npCol)


	return np.matrix(npCols).T



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
		return chr(48 + num) # = str(num)
	else:
		return chr(55 + num)

