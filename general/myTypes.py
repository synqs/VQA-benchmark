import networkx as nx

from qiskit.circuit import Parameter
from qiskit.providers.basebackend import BaseBackend
from qiskit import QuantumCircuit

from scipy.optimize.optimize import OptimizeResult


from typing import List, Dict, Tuple, Iterable, Sequence, Mapping, Callable
from typing import Union, Optional, Any, cast, Type#, NoReturn
from numpy.typing import ArrayLike, NDArray
from quantum.thetas import Parameters



Number			= Union[int, float]

AllOptions		= Dict
Options			= Dict
Solution		= Optional[Tuple[List[str], Number]]
Result			= Tuple[Optional[Number], Number, Optional[Number], float]

