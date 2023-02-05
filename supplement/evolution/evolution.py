#!/usr/bin/env python3.8
# -*- coding: utf8 -*-

import numpy as np
from numpy import linalg
from typing import Tuple, List, Callable, NoReturn
import sympy
from scipy import sparse
import time as clock
from itertools import permutations


def adjacency(n: int, basis: List[Tuple]) -> sympy.SparseMatrix: # , top_down: bool = False
	nfac = len(basis)
	# nfac = basis(n)

	nonzero = []
	for k in range(nfac):
		for j in range(k):
			differences = 0
			for m in range(n):
				different = basis[j][m] != basis[k][m]
				differences += int(different)
				if differences > 2:
					break
			if differences == 2:
				nonzero.append((j, k))
				nonzero.append((k, j))

	array = np.zeros((nfac, nfac), dtype=int)
	for item in nonzero:
		array[item] = 1
	data = {item:1 for item in nonzero}


	return array # sympy.SparseMatrix(nfac, nfac, data)

def polynomial(linear: sympy.SparseMatrix, exponent: int) -> sympy.SparseMatrix:

	matrix = sympy.banded(linear.shape[0], {0:1})

	results = [matrix]

	for _ in range(exponent+1):
		matrix = linear @ matrix
		results.append(matrix)
	
	return results

def hamilton(n: int, adj: sympy.SparseMatrix) -> sympy.SparseMatrix:
	return - adj / (int(round(n * (n-1) / 2))) # negative for simpler minimum

def unitary_matr(ham: sympy.SparseMatrix) -> sympy.SparseMatrix:
	x = sympy.Symbol("x") # \lambda = λ

	return sympy.simplify(sympy.exp(- sympy.I * x * ham))

def unitary_func(basis: List[Tuple], uni: sympy.SparseMatrix) -> Callable[[float], sparse.csr_matrix]:
	nfac = len(basis)

	def fill(theta: float) -> sparse.csr_matrix:
		expression = Unitary.subs(x, theta)
		dictionary = expression.evalf().todok()

		return sparse.csr_matrix(([complex(value) for value in dictionary.values()], zip(*dictionary.keys())), shape=(nfac, nfac))

	return fill


def express(M: sympy.SparseMatrix, timing: float = 0.) -> NoReturn:
	shape  = M.shape
	values = set(M.flat)
	# values = set(M.values())
	size_v = len(values)
	indics = M==0
	nozero = np.sum(indics == False)
	# nozero = M.nnz()
	# indics.flags.writeable = False
	# indics = sympy.ImmutableSparseMatrix(shape[0], shape[1], {item:1 for item in M.todok().keys()})
	hashed = "{:0x}".format(hash(indics.data.tobytes()) % 16**6)

	print(hashed, shape, size_v, nozero, timing, values)


def polynomial_scaling(n: int, kmax: int) -> int:
	print('Calculate all permutations...')
	basis = list(permutations(range(n)))

	print('Calculate adjacency matrix...')
	before = clock.time()
	A = adjacency(n, basis)
	runtime = clock.time() - before
	print(f'Calculated in {runtime:.3f} s.')
	
	# matrix = sympy.banded(A.shape[0], {0:1})
	matrix = np.identity(A.shape[0], dtype=int)
	express(matrix)
	for k in range(kmax + 1):
		before = clock.time()
		matrix = matrix @ A
		runtime = clock.time() - before
		express(matrix, runtime)



def decompose_even(n: int, kmax: int) -> int:
	print('Calculate all permutations...')
	basis = list(permutations(range(n)))

	print('Calculate adjacency matrix...')
	before = clock.time()
	A = adjacency(n, basis)
	runtime = clock.time() - before
	print(f'Calculated in {runtime:.3f} s.')
	

	A2 = A @ A





	# matrix = sympy.banded(A.shape[0], {0:1})
	matrix = np.identity(A.shape[0], dtype=int)
	even = [((0,0), matrix)]
	for k in range(kmax + 1):
		before = clock.time()
		matrix = matrix @ A2
		after = clock.time()

		rest_of_matrix = matrix
		decomposition = []
		for position, vector in even:
			component = rest_of_matrix[position]
			decomposition.append(component)
			rest_of_matrix -= component * vector
		for i in range(matrix.shape[1]):
			if rest_of_matrix[0,i] != 0:
				component = rest_of_matrix[0,i]
				candidate = rest_of_matrix / component
				# assert (candidate == 0 or candidate == 1).all()
				even.append(np.int(np.round(candidate)))
				decomposition.append(component)
				rest_of_matrix -= component * candidate
		assert (rest_of_matrix == 0).all(), rest_of_matrix
		runtime = clock.time() - before
		print(decomposition)




runtimes = []
for i in range(10):
	kmax = i + 5
	print(f"Start process for {i} objects up to order {kmax}.")
	before = clock.time()
	polynomial_scaling(i, kmax)
	runtime = clock.time() - before
	print(f"Finished process for {i} objects up to order {kmax} in {runtime:.3f} s.\n")
	runtimes.append(runtime)

print(runtimes)
