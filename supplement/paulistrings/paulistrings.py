#!/usr/bin/env python3.8
# -*- coding: utf8 -*-

import numpy as np
from numpy import linalg


nodes: int = 3
n: int = nodes * nodes
l: int = 2**n

symmetric_number: int = 0 # = 73 = 0b001001001 for nodes = 3
for i in range(nodes):
	symmetric_number <<= nodes
	symmetric_number  +=   1

def has_symmetry(a: int, n: int) -> bool:
	string = format(a, f'0{n*n}b')
	for j in range(n):
		for i in range(j):
			# print(i*n,(i+1)*n, j*n,(j+1)*n)
			if string[i*n:(i+1)*n] == string[j*n:(j+1)*n]:
				return True
	return False

a: np.array = np.empty(l, dtype=np.int32)
for i in range(l):
	if i % symmetric_number == 0: # 73
		a[i] = 0
	elif has_symmetry(i, nodes):
		a[i] = 2
	else:
		a[i] = 3

a_frequencies = {}
for number in a:
	try:
		a_frequencies[number] += 1
	except KeyError:
		a_frequencies[number]  = 1
print(a_frequencies)

int_hadamard : np.array = np.array([1])
# int_hadmrd_1 : np.array = np.array([[1,1], [1,-1]])
for _ in range(n):
	int_hadamard = np.block([[int_hadamard, int_hadamard], [int_hadamard, -int_hadamard]])
	# int_hadamard = np.kron(int_hadmrd_1, int_hadamard)

# b = np.zeros(l)
# b[3] = 1

result : np.array = int_hadamard @ a
print(np.array(result / 64, dtype=np.int16)) # / 3 / 2**(n-6)

for i in range(l):
	if result[i] == 0:
		continue
	print(format(i, '3d'), format(i, f'0{n}b'), format(i, f'0{n}b').replace('0', 'I').replace('1', 'Z'), f'{int(result[i]/64)}/24')
