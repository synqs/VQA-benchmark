import itertools
import numpy as np
from math import factorial
from scipy import sparse
import sympy

import networkx as nx
import time as clock

from typing import Tuple, List, Sequence, Callable
from numpy.typing import NDArray


def tsp_problem_hamiltonian(n: int, basis: List[Tuple], G: nx.Graph, full: bool) -> NDArray:
	nfac = len(basis)

	diagonals = []
	for perm in basis:
		diagonal = 0
		for nFrom, nTo in zip(perm[:-1], perm[1:]):
			diagonal += G[nFrom][nTo]['weight']
		if full:
			diagonal += G[perm[-1]][perm[0]]['weight']
		else:
			diagonal += G[perm[-1]][n-1]['weight'] + G[n-1][perm[0]]['weight']
		diagonals.append(diagonal)

	assert len(diagonals) == nfac

	return np.array(diagonals)


def tsp_problem_unitary(diagonals: NDArray) -> Callable[[float], sparse.csr_matrix]:
	nfac = len(diagonals)
	x = sympy.Symbol("λ")

	rotations = [sympy.exp(- sympy.I * x * element) for element in diagonals]

	# Unitary = sympy.SparseMatrix(nfac, nfac, {(i,i):rotations[i] for i in range(nfac)})

	def fill(theta: float) -> sparse.csr_matrix:
		return sparse.csr_matrix(([complex(rotation.subs(x, theta)) for rotation in rotations], (range(nfac), range(nfac))), shape=(nfac, nfac))

	return fill



def tsp_mixer_unitary(n: int, basis: List[Tuple]) -> Tuple[NDArray, NDArray]: # , top_down: bool = False
	# for permutation in itertools.permutations(range(n)):
	# 	perm = np.array(permutation)
	# 	basis.append(perm)
	# 
	nfac = len(basis)
	# nfac = basis(n)

	Hamiltonian: NDArray = np.zeros((nfac, nfac))

	# nonzero = []
	differences: int
	for k in range(nfac):
		for j in range(k):
			differences = 0
			for num_j, num_k in zip(basis[j], basis[k]):
				# different = num_j != num_k
				differences += int(num_j != num_k)
				if differences > 2:
					break
			if differences == 2:
				Hamiltonian[j,k] = - 1
				Hamiltonian[k,j] = - 1
				# nonzero.append((j, k))
				# nonzero.append((k, j))

	# print(Hamiltonian)

	# data = {item:1 for item in nonzero}

	# x = sympy.Symbol("x") # \lambda = λ

	# Hamiltonian = - sympy.SparseMatrix(nfac, nfac, data) / (int(round(n * (n-1) / 2))) # negative for simpler minimum
	# Unitary     =   sympy.simplify(sympy.exp(- sympy.I * x * Hamiltonian))

	# def fill(theta: float) -> sparse.csr_matrix:
	# 	expression = Unitary.subs(x, theta)
	# 	dictionary = expression.evalf().todok()

	# 	return sparse.csr_matrix(([complex(value) for value in dictionary.values()], zip(*dictionary.keys())), shape=(nfac, nfac))

	# return fill



	# Ssum = 0
	# Rsum = 0
	# for k2 in range(1,n+1):
	# 	for k1 in range(k2):
	# 		T = np.identity(n, dtype=np.int8)
	# 		T[k1][k1] = 0
	# 		T[k2][k2] = 0
	# 		T[k1][k2] = 1
	# 		T[k2][k1] = 1

	# 		# print(T)

	# 		S = np.zeros((nfac, nfac), dtype=np.int8)
	# 		R = np.zeros((nfac, nfac), dtype=np.int8)
	# 		for i in range(nfac):
	# 			for j in range(nfac):
	# 				if (basis[i] == T @ basis[j]).all(): # TODO is this the correct and consistent implementation?
	# 					S[i][j] = 1
	# 				if (basis[i] == basis[j] @ T).all(): # TODO is this the correct and consistent implementation?
	# 					R[i][j] = 1
	# 		# print(S)
	# 		Ssum += S
	# 		Rsum += R
	# # print(Ssum)
	
	# # np.set_printoptions(edgeitems=30, linewidth=100000, 
	# # formatter=dict(float=lambda x: "%5.2f" % x))

	# Sews, Sevs = np.linalg.eigh(Ssum)
	# Rews, Revs = np.linalg.eigh(Rsum)
	# # ew_dict = {}
	# # # for i in range(len(ews)):
	# # 	# print(i+1, round(ews[i], 1))
	# # for ew_full in ews:
	# # 	ew = round(ew_full, 1)
	# # 	try:
	# # 		ew_dict[ew] += 1
	# # 	except KeyError:
	# # 		ew_dict[ew]  = 1
	# # print(ew_dict)
	# # print(evs)
	# # print(np.round(evs, 2))
	# # H = np.matrix(Ssum)
	# # print((H - H.T).any())
	# # np.savetxt('eigenvectors.csv', evs, delimiter=';')
	# # U = np.matrix(evs)
	# # print(np.abs(U @ U.T - np.identity(nfac)) > .000001)
	# # print(np.abs(U @ np.diag(ews) @ U.T - H) > .000001)

	# # from testing we know:
	# # Ssum = Ssum.T = Ssum.H
	# # evs.T = evs.H = evs^-1
	# # evs @ diag(ews) @ evs.T = Ssum
	
	start: float = clock.time()
	ews, evs = np.linalg.eigh(Hamiltonian)
	end:   float = clock.time()

	return ews, evs, end - start
	# # return Rews, Revs


# minimum of the negative Hamiltonian
def tsp_initial(n: int, basis: List[Tuple]) -> NDArray:
	nfac = factorial(n)
	return np.ones(nfac) / np.sqrt(nfac)


# def tsp_initial(n: int, basis: List[Tuple]) -> NDArray:
# 	nfac = factorial(n)
# 	signs = []
# 	for perm in basis:
# 		sign = 1
# 		for j in range(n):
# 			for i in range(j):
# 				sign *= (perm[j] - perm[i]) / (j - i) # TODO ? very inefficient implementation -> https://math.stackexchange.com/questions/65923/how-does-one-compute-the-sign-of-a-permutation or https://code.activestate.com/recipes/578227-generate-the-parity-or-sign-of-a-permutation/
# 		signs.append(np.round(sign))
# 	return np.array(signs) / np.sqrt(nfac)




def run(thetas: Sequence[float], Hmix: Tuple[NDArray, NDArray], Hprob: NDArray, init: NDArray, p: int) -> NDArray:
	# Umix @ Uprob @ ... @ Uprob @ init
	# Hmix = Vmix @ np.diag(Λmix) @ Vmix.T

	# M @ np.diag(D) = M * D for D.shape = (n,)
	current = init
	Λmix, Vmix = Hmix
	iHprob = - 1j * Hprob
	iΛmix  = - 1j * Λmix

	for i in range(p):
		current = Vmix.T @ (np.exp(iHprob * thetas[2*i  ]) * current)
		current = Vmix   @ (np.exp(iΛmix  * thetas[2*i+1]) * current)
		# current = Uprob(thetas[2*i  ]) @ current
		# current = Umix (thetas[2*i+1]) @ current
	
	return current


def get_expectation_value(state: NDArray, Hprob: NDArray) -> float:
	scalar_product = np.vdot(state, Hprob * state)
	assert np.isclose(np.imag(scalar_product), 0, atol= 10**(-6)), "non-hermitian Hprob"
	return np.real(scalar_product)