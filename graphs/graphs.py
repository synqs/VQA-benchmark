import networkx as nx

from typing import List, Tuple, Union, Optional, Any

from general.myTypes import Solution


def get(size: str, problem: str, distances: Any) -> Tuple[nx.Graph, int, Solution]:
	graph: nx.Graph
	n: int
	weight: List[List[Union[int, float]]]
	solution: Solution
	if size == "tiny":
		graph = nx.Graph()
		graph.add_node(0)
		graph.add_node(1)
		graph.add_edge(0, 1, weight=1)
		nx.freeze(graph)
		n = 2
		solution = ["01"], -1 if problem == "max_cut" else +2
	elif size == "small":
		n, weights, solution = select_weights(problem, distances)
		graph = get_graph_from_weights(n, weights)
	elif size == "large":
		from .landkarte import full_netz_id, solutions
		graph = full_netz_id
		n = len(graph)
		solution = solutions[problem]
	else:
		raise KeyError("Unknown problem '"+ problem +"'.")
	
	return graph, n, solution


def select_weights(problem: str, distances: Any) -> Tuple[int, List[List[Union[int, float]]], Solution]:
	n: int = 4
	weights: List[List[Union[int, float]]]
	solution: Solution
	if problem == "max_cut":
		weights = [ [0, 0, 0, 0],
					[1, 0, 0, 0],
					[2, 3, 0, 0],
					[4, 5, 6, 0]]
		solution = ["0001"], -15

		if distances == 2:
			weights = [ [0, 0, 0, 0],
						[1, 0, 0, 0],
						[3, 2, 0, 0],
						[6, 5, 4, 0]]
			solution = ["0011"], -16
	elif problem == "tsp":
		weights = [ [0, 0, 0, 0],
					[5, 0, 0, 0],
					[1, 4, 0, 0],
					[2, 1, 3, 0]]
		solution = ['0213', '0312'], -8
	else:
		raise KeyError("Unknown problem '"+ problem +"'.")
	return n, weights, solution


def get_graph_from_weights(graph_size: int, weights: List[List[Union[int, float]]]) -> nx.Graph:
	G: nx.Graph = nx.Graph()
	for i in range(graph_size):
		G.add_node(i)
		for j in range(i):
			G.add_edge(i, j, weight=weights[i][j])
	
	nx.freeze(G)
	return G


