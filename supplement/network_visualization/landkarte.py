#!/usr/bin/env python3.9
# -*- coding:utf-8 -*-

# Usage:
# python3 landkarte.py; convert -crop 850x1100+190+260 Map.png Map.png; convert -crop 850x1100+190+260 Route.png Route.png
# # python3 landkarte.py; convert -crop 480x640+60+80 Map.png Map.png; convert -crop 480x640+60+80 Route.png Route.png
# # from landkarte import netz, full_netz, draw_graph



# For the graph
import networkx as nx
# from graph import Graph


# Basic libraries
import matplotlib.pyplot as plt 								# for plotting
# from typing import NamedTuple									# to transform Town to dict
import numpy as np  											# for coordinate array




class Town():
	def __init__(self, label, coordinateN, coordinateE, name = None):
		if name is None:
			name = label
		self.label  = label
		self.id     = None
		# self.id     = str_to_int(label)
		# self.coords = [coordinateN, coordinateE]
		self.coords = np.array([coordinateE, coordinateN])
		# self.coN = coordinateN
		# self.coE = coordinateE
		self.name = name

	def __str__(self):
		return f"{self.label} ({self.name}: {self.co[1]:.2f}° N, {self.co[0]:.2f}° E)"
	
	def set_id(self, new_id):
		self.id = new_id

	def asdict(self):
		return {
			'id':     self.id,
			'label':  self.label,
			'coords': self.coords,
			'name':   self.name,
			# 'color':  '#48f',
			}

towns = []
lines = []

# m = Town("M", 48.133333333333333, 11.583333333333333, "München")

towns.append(Town('HH', 53.55, 10, 'Hamburg'))
# towns.append(Town('HB', 53.0833333333333, 8.8, 'Bremen'))
towns.append(Town('B', 52.5166666666667, 13.4, 'Berlin'))
towns.append(Town('H', 52.3666666666667, 9.73333333333333, 'Hannover'))
towns.append(Town('DO', 51.5166666666667, 7.46666666666667, 'Dortmund'))
# towns.append(Town('E', 51.45, 7.01666666666667, 'Essen'))
towns.append(Town('L', 51.3333333333333, 12.3666666666667, 'Leipzig'))
# towns.append(Town('D', 51.2333333333333, 6.78333333333333, 'Düsseldorf'))
towns.append(Town('K', 50.9333333333333, 6.95, 'Köln'))
towns.append(Town('F', 50.1166666666667, 8.68333333333333, 'Frankfurt am Main'))
towns.append(Town('WÜ', 49.8, 9.93333333333333, 'Würzburg'))
towns.append(Town('MA', 49.4833333333333, 8.46666666666667, 'Mannheim'))
towns.append(Town('N', 49.45, 11.0833333333333, 'Nürnberg'))
towns.append(Town('S', 48.7833333333333, 9.18333333333333, 'Stuttgart'))
towns.append(Town('M', 48.1333333333333, 11.5833333333333, 'München'))
towns.append(Town('FR', 48, 7.85, 'Freiburg im Breisgau'))
# towns.append(Town('KS', 51.3167, 9.5, 'Kassel'))
# towns.append(Town('UL', 48.4, 9.9833, 'Ulm'))




# Referenz: Direktverbindung mit ICE, ab 21.12.22, 11 Uhr, Hbf (außer Kassel), kaufmännisch gerundet, kürzere Richtung bei deutlichen Unterschieden
lines.append(['HH', 'DO', 17])
lines.append(['DO', 'K' ,  7])
lines.append([ 'K', 'F' ,  7])
lines.append([ 'F', 'MA',  4])
lines.append(['MA', 'FR',  9])
lines.append(['MA', 'S' ,  4])
lines.append([ 'S', 'M' , 12])
lines.append([ 'M', 'N' ,  6])
lines.append([ 'L', 'N' , 12])
lines.append([ 'S', 'N' , 13]) # IC
lines.append([ 'N', 'WÜ',  5])
lines.append(['WÜ', 'H' , 12])
lines.append(['HH', 'H' ,  7])
lines.append(['HH', 'B' , 11])
lines.append([ 'B', 'L' ,  7])
lines.append([ 'L', 'WÜ', 15]) # deutlich kürzer als umgekehrt
lines.append(['WÜ', 'F' ,  7])
lines.append(['DO', 'H' , 10])
lines.append([ 'B', 'H' , 10])
# lines.append([ 'S', 'FR', 11])
lines.append([ 'F', 'H' , 14])
lines.append([ 'H', 'L' , 16])
# lines.append(['HB', 'H' ,  6])
# lines.append(['HB', 'HH',  6])
# lines.append(['HB', 'DO', 11])
# lines.append(['KS', 'WÜ',  7])
# lines.append(['KS', 'H' ,  5])
# lines.append(['KS', 'F' ,  8])
# lines.append(['UL', 'M' ,  8])
# lines.append(['UL', 'S' ,  6])
# lines.append(['UL', 'N' , 12])
# lines.append(['UL', 'WÜ', 17])
# lines.append(['MA', 'WÜ',  6])

str_to_int = {}
for i in range(len(towns)):
	towns[i].set_id(i)
	str_to_int[towns[i].label] = i



netz    = nx.Graph()
netz_id = nx.Graph()

for town in towns:
	# netz.add_node(town.label, town)
	netz.add_node(town.label, **town.asdict())
	netz_id.add_node(town.id, **town.asdict())

for line in lines:
	s, e, d = line
	# netz.add_edge(s, e, value = d, bidirectional=True)
	netz.add_edge(s, e, weight = d, color = 'k', draw = True, special = False)
	# netz.add_edge(str_to_int(s), str_to_int(e), weight = d)
	netz_id.add_edge(str_to_int[s], str_to_int[e], weight = d, color = 'k', draw = True, special = False)


full_netz = nx.Graph()
full_netz.add_nodes_from(netz.nodes(data=True))

full_netz_id = nx.Graph()
full_netz_id.add_nodes_from(netz_id.nodes(data=True))

selected_towns = ['DO', 'L', 'F', 'WÜ', 'N', 'S']

tiny_netz    = nx.Graph()
tiny_netz_id = nx.Graph()

str_to_tiny_int = {}
for town in selected_towns:
	tiny_netz.add_node(town, **netz.nodes()[town])

for i in range(len(selected_towns)):
	tiny_netz.nodes()[selected_towns[i]]['id'] = i
	str_to_tiny_int[selected_towns[i]] = i


for town in selected_towns:
	tiny_netz_id.add_node(str_to_tiny_int[town], **tiny_netz.nodes()[town])

for start in netz.nodes():
	for end in netz.nodes():
		if end == start:
			break
		#d, 
		p = nx.shortest_path(netz, start, end, weight='weight')
		d = 0
		for pair in zip(p[:-1], p[1:]):
			s, e = pair
			d += netz[s][e]['weight']
		# print(d)
		h = int(d / 6)
		t = f"{h:d}:{d-h*6:d}0"
		if h:
			t = f"{h:d}:{d-h*6:d}0"
		else:
			t = f":{d-h*6:d}0"
		t = str(10*d)
		full_netz.add_edge(start, end, weight = d, label = t, calculated = True, color = 'w', draw = False, special = False) #, bidirectional=True)
		full_netz_id.add_edge(str_to_int[start], str_to_int[end], weight = d, label = t, calculated = True, color = 'w', draw = False, special = False)
		if start in tiny_netz and end in tiny_netz:
			tiny_netz.add_edge(start, end, weight = d, label = t, calculated = True, color = 'k', draw = True, special = False)
			tiny_netz_id.add_edge(str_to_tiny_int[start], str_to_tiny_int[end], weight = d, label = t, calculated = True, color = 'k', draw = True, special = False)



for edge in netz.edges():
	s, e = edge
	full_netz[s][e]['calculated'] = False
	full_netz[s][e]['color'] = 'k'
	full_netz[s][e]['draw']  = True
	# full_netz[s][e]['label'] = full_netz[s][e]['weight']
	# assert full_netz[s][e]['weight'] == netz[s][e]['weight'], edge
	if (s, e) in tiny_netz.edges():
		tiny_netz[s][e]['calculated'] = False
		tiny_netz[s][e]['color'] = 'k'
		tiny_netz[s][e]['draw']  = True
		# tiny_netz[s][e]['label'] = tiny_netz[s][e]['weight']

for edge in netz_id.edges():
	s, e = edge
	full_netz_id[s][e]['calculated'] = False
	full_netz_id[s][e]['color'] = 'k'
	full_netz_id[s][e]['draw']  = True
	# full_netz_id[s][e]['label'] = full_netz_id[s][e]['weight']
	# assert full_netz_id[s][e]['weight'] == netz_id[s][e]['weight'], edge
	if (s, e) in tiny_netz_id.edges():
		tiny_netz_id[s][e]['calculated'] = False
		tiny_netz_id[s][e]['color'] = 'k'
		tiny_netz_id[s][e]['draw']  = True
		# tiny_netz_id[s][e]['label'] = tiny_netz_id[s][e]['weight']


def draw_graph(G, fname):
	node_label_attribute = 'label'
	node_pos_attribute   = 'coords'
	# node_label_attribute = 'label'
	edge_draw_attribute  = 'draw'
	edge_label_attribute = 'label'
	# edge_label_attribute = 'color'

	edges   = [edge for edge, draw in nx.get_edge_attributes(G, edge_draw_attribute).items() if draw]
	if not edges:
		edges = G.edges

	node_labels = nx.get_node_attributes(G, node_label_attribute)
	node_coords = nx.get_node_attributes(G, node_pos_attribute)
	# node_colors = nx.get_node_attributes(G, 'color').values()
	# edge_draw   = nx.get_edge_attributes(G, 'draw')#, default=True)
	# edge_draw   = {'job[%s]' % k if k != 'id' else k: v for k, v in nx.get_edge_attributes(G, 'draw', default=True).items()}
	edge_labels = nx.get_edge_attributes(G, edge_label_attribute)
	try:
		try:
			edge_colors = ['r' if G[s][e]['special'] else G[s][e]['color'] for s,e in edges]# nx.get_edge_attributes(G, 'color').values() # ('r' if special else 'b')
		except KeyError:
			edge_colors = [G[s][e]['color'] for s,e in edges]# nx.get_edge_attributes(G, 'color').values() # ('r' if special else 'b')
	except KeyError:
		edge_colors = 'k'
	



	# plt.figure(figsize=(4,6))
	plt.figure(figsize=(12, 16))
	default_axes = plt.axes(frameon=False)
	common_parameters = dict(G=G, ax=default_axes, pos=node_coords)
	nx.draw_networkx(**common_parameters, edgelist=edges, labels=node_labels, node_size=1760, font_size=22, edge_color=edge_colors, node_color='#8FBCDA')
	common_parameters['font_size'] = 18 # for all visible edges
	# nx.draw_networkx_edges(**common_parameters, edgelist=edges, labels=node_labels, edge_color=edge_colors, node_color='#8FBCDA')
	if edge_colors == 'k':
		nx.draw_networkx_edge_labels(**common_parameters, edge_labels={(s,e):G[s][e][edge_label_attribute] for s,e in edges})
	else:
		for color in set(edge_colors):
			nx.draw_networkx_edge_labels(**common_parameters, edge_labels={(s,e):G[s][e][edge_label_attribute] for s,e in edges if G[s][e]['color'] == color}, font_color=color)
			# print('Draw '+str(len([s for s,e in edges if G[s][e]['color'] == color]))+f' edges in "{color}".')
			# nx.draw_networkx_edge_labels(**common_parameters, edge_labels={(s,e):G[s][e][edge_label_attribute] for s,e in edges if G[s][e]['color'] == color or (G[s][e]['special'] and color == 'r')}, font_color=color)
		try:
			nx.draw_networkx_edge_labels(**common_parameters, edge_labels={(s,e):G[s][e][edge_label_attribute] for s,e in edges if G[s][e]['special']}, font_color='r')
		except KeyError:
			pass
	plt.savefig(fname)


def reset_graph(G):
	for s,e in G.edges:
		G[s][e]['special'] = False
		try:
			if G[s][e]['draw'] == 'temp':
				G[s][e]['draw'] = False
		except KeyError:
			pass

def draw_path(G, order, name):
	try:
		rotate = order[1:]+(order[0],)
	except TypeError:
		rotate = order[1:]+[order[0],]
	for s,e in zip(order, rotate):
		G[s][e]['special'] = True
		try:
			if not G[s][e]['draw']:
				G[s][e]['draw'] = 'temp'
		except KeyError:
			pass
	draw_graph(G, name)
	reset_graph(G)



tsp = ['HH', 'B', 'L', 'WÜ', 'N', 'M', 'S', 'FR', 'MA', 'F', 'K', 'DO', 'H'] # correct solution
solutions = {
	"TSP":			(["01479BAC86532", "023568CAB9741"], + 123), # test the number again!
	"TSP_full":		(['023568CAB9741', '23568CAB97410', '3568CAB974102', '568CAB9741023', '68CAB97410235', '8CAB974102356', 'CAB9741023568', 'AB9741023568C', 'B9741023568CA', '9741023568CAB', '741023568CAB9', '41023568CAB97', '1023568CAB974', '01479BAC86532', '1479BAC865320', '479BAC8653201', '79BAC86532014', '9BAC865320147', 'BAC8653201479', 'AC8653201479B', 'C8653201479BA', '8653201479BAC', '653201479BAC8', '53201479BAC86', '3201479BAC865', '201479BAC8653'], + 123), # test the number again!
	"MCP":		(["1111111000000"], - 991), # both not proven!
	"MCP_full":	(["1111111000000", "0000000111111"], - 991), # both not proven!
}

# tsp = ['DO', 'L', 'N', 'WÜ', 'F', 'S', 'DO'] # correct solution
tiny_solutions = {
	"TSP":			(["014325", "052341"], + 103),
	"TSP_full":		(['014325', '143250', '432501', '325014', '250143', '501432', '052341', '523410', '234105', '341052', '410523', '105234'], + 103),
	"MCP":		(["001101"], - 179),
	"MCP_full":	(["001101", "110010"], - 179),
}


nx.freeze(netz)
nx.freeze(netz_id)
nx.freeze(full_netz)
nx.freeze(full_netz_id)
nx.freeze(tiny_netz)
nx.freeze(tiny_netz_id)

draw_graph(full_netz, "Map.png")
draw_path(full_netz, tsp, "Route.png")

