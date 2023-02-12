from qiskit.circuit import Parameter

from sympy import Symbol

from typing import List, Union

class Parameters_qiskit:
	_thetas: List[Parameter] = []

	_length: int
	_countr: int

	def __init__(self, intended_length: int):
		self._length = intended_length
		self._countr = 0
		self._thetas = []
		for i in range(intended_length):
			self._thetas.append(Parameter(f'$\\theta_{{{i}}}$'))
			# if i % 2:
			# 	self._thetas.append(Parameter(f'$\\beta_{{{int(i/2)}}}$'))
			# else:
			# 	self._thetas.append(Parameter(f'$\\gamma_{{{int(i/2)}}}$'))

	def __len__(self):
		assert self._length == len(self._thetas), "self._length = "+ self._length +", but len(self._thetas) = "+ len(self._thetas)
		return self._length

	def __iter__(self):
		return self
	def __next__(self) -> Parameter:
		if self._countr >= self._length:
			raise KeyError("Too many Parameters used")
		n: Parameter = self._thetas[self._countr]
		self._countr += 1
		return n
	
	def is_complete(self):
		return len(self) == self._countr

	
class Parameters_sympy:
	_thetas: List[Symbol]

	_length: int
	_countr: int

	def __init__(self, intended_length: int):
		self._length = intended_length
		self._countr = 0
		self._thetas = []
		for i in range(intended_length):
			self._thetas.append(Symbol(f'$\\theta_{{{i}}}$'))

	def __len__(self):
		assert self._length == len(self._thetas), "self._length = "+ self._length +", but len(self._thetas) = "+ len(self._thetas)
		return self._length

	def __iter__(self):
		return self
	def __next__(self) -> Symbol:
		if self._countr >= self._length:
			raise KeyError("Two many Parameters used")
		n: Symbol = self._thetas[self._countr]
		self._countr += 1
		return n
	
	def is_complete(self):
		return len(self) == self._countr

	
Parameters = Union[None, Parameters_qiskit, Parameters_sympy]




# def submit(thetas):
# 	results.append(thetas)
# 	reset()

# def get_results():
# 	return results