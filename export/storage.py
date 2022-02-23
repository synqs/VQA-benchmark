import os

from typing import Mapping


storage_folder: str = "storage" + os.sep # ".." +  os.sep +

folders: Mapping[str, str] = {
	"data":		storage_folder + "data" + os.sep,
	"circ":		storage_folder + "img"  + os.sep + "circuit" + os.sep,
	"dist":		storage_folder + "img"  + os.sep + "distribution" + os.sep,
	"comp":		storage_folder + "img"  + os.sep + "compare" + os.sep,
	# "runt": 	storage_folder + "img"  + os.sep + "runtimes" + os.sep,
	"thta":		storage_folder + "data" + os.sep + "img" + os.sep + "optima" + os.sep,
}


files: Mapping[str, str] = {
	# "data":					storage_folder + "data" + os.sep,
	"circuit_thetas":		folders["circ"] + "{qubase}-{qAlgorithm}-{problem}-{size}-{p}-thetas.png",
	"circuit_values":		folders["circ"] + "{qubase}-{qAlgorithm}-{problem}-{size}-{p}-values.png",
	"distributions":		folders["dist"] + "{qubase}-{qAlgorithm}-{problem}-{size}-{p}.png",
	"comparisons":			folders["comp"] + "{qubase}-{qAlgorithm}-{problem}-{size}-{pmax}.png",
	# "runtimes": 			folders["runt"] + "{qubase}-{qAlgorithm}-{problem}-{size}-{pmax}.png",
	"optimal_thetas":		folders["thta"] + "{qubase}-{qAlgorithm}-{problem}-{size}-{p}",
}


