import networkx as nx

from qiskit.circuit import Parameter
try:
	from qiskit.providers.basebackend import BaseBackend
except ModuleNotFoundError:
	from qiskit.providers.backend import Backend as BaseBackend
from qiskit import QuantumCircuit

from scipy.optimize.optimize import OptimizeResult
from scipy.sparse import csr_matrix as csr


from typing import List, Dict, Tuple, Iterable, Sequence, Mapping, Callable
from typing import Union, Optional, Any, cast, Type#, NoReturn
from numpy.typing import ArrayLike, NDArray
from quantum.thetas import Parameters
# from sympy import Symbol, SparseMatrix


Number			= Union[int, float]

AllOptions		= Dict
Options			= Dict
Solution		= Optional[Tuple[List[str], Number]]
Result			= Tuple[Optional[Number], Number, Optional[Number], float]

