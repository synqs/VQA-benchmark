from qiskit.circuit import Parameter

from typing import List


class Parameters:
	_thetas: List[Parameter] = []

	_length: int
	_countr: int

	def __init__(self, intended_length: int):
		self._length = intended_length
		self._countr = 0
		self._thetas = []
		for i in range(intended_length):
			self._thetas.append(Parameter(f'$\\theta_{{{i}}}$'))

	def __len__(self):
		assert self._length == len(self._thetas), "self._length = "+ self._length +", but len(self._thetas) = "+ len(self._thetas)
		return self._length

	def __iter__(self):
		return self
	def __next__(self) -> Parameter:
		if self._countr >= self._length:
			raise KeyError("Two many Parameters used")
		n: Parameter = self._thetas[self._countr]
		self._countr += 1
		return n
	
	def is_complete(self):
		return len(self) == self._countr

	





# def submit(thetas):
# 	results.append(thetas)
# 	reset()

# def get_results():
# 	return results