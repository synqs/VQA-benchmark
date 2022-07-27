from qutip import Qobj

from sympy import Symbol

from typing import List, Union, NoReturn

class FakeCircuit:
	_operators: List[Qobj]

	def __init__(self):
		self._operators = []

	def __len__(self):
		return len(self._operators)

	def draw(self):
		return False
	
	def run(self) -> List:
		return _operators
	
	def append(self, new_item: Qobj) -> NoReturn:
		_operators.append(new_item)

	
class Parameters_sympy:
	_thetas: List[Symbol] = []

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

	
Parameters = Union[Parameters_qiskit, Parameters_sympy]




# def submit(thetas):
# 	results.append(thetas)
# 	reset()

# def get_results():
# 	return results