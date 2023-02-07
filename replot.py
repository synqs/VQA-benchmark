#!/usr/bin/env python3.9
# -*- coding:utf-8 -*-

import export.comparison
# import quantum.circuit

from export.storage import folders


import os
import numpy as np


folder: str = folders['comp'] # overview plots
for image in os.scandir(folder):
	if image.is_dir():
		continue
	reports, maxP, _ = np.load(image.path.replace("img", "data"+ os.sep +"img").replace(".png", "_all.npy"), allow_pickle=True) # _ = fileName
	# # print(folder + image, 'reports', maxP, fileName)
	# exceptions = ['allqAlgorithm-MCP-small-4.png', 'allqAlgorithm-MCP-tiny-4.png', 'VQEs-MCP-large-4.png']
	# assert fileName == image.path or image.name in exceptions, (fileName, folder, image.name)
	# imgTime = - 1645000000 + os.path.getmtime(image.path)
	# datTime = - 1645000000 + os.path.getmtime(image.path.replace("img", "data"+ os.sep +"img").replace(".png", "_all.npy"))

	if image.name == 'allqAlgorithm-MCP-tiny-4.png':
		reports = reversed(reports)
	# print(imgTime, datTime, imgTime - datTime)
	# # if abs(imgTime - datTime) > 1:
	# # 	print(image.path, image.path.replace("img", "data"+ os.sep +"img").replace(".png", "_all.npy"))
	# # print()

	varied = reports[0][2]
	# print(varied)
	export.comparison.plot(reports, maxP, image.path)
	


# folder: str = folders['dist'] # histograms
# for image in os.scandir(folder):
# 	if image.is_dir():
# 		continue
# 	if not (os.path.exists(image.path.replace("img", "data"+ os.sep +"img").replace(".png", ".npy")) and os.path.exists(image.path.replace("img", "data"+ os.sep +"img").replace(".png", "_sol.npy"))):
# 		# print(f"File {image.name.replace('.png', '_all.npy')} doesn't exist")
# 		continue
	
# 	print("Actually doing something")
# 	states, rates = np.load(image.path.replace("img", "data"+ os.sep +"img").replace(".png",     ".npy"), allow_pickle=True)
# 	topStat, minE = np.load(image.path.replace("img", "data"+ os.sep +"img").replace(".png", "_sol.npy"), allow_pickle=True)
	
# 	counts = {k:v for k,v in zip(states, rates)}


# 	# imgTime = - 1645000000 + os.path.getmtime(image.path)
# 	# datTime = - 1645000000 + os.path.getmtime(image.path.replace("img", "data"+ os.sep +"img").replace(".png", ".npy"))
# 	# datTime = - 1645000000 + os.path.getmtime(image.path.replace("img", "data"+ os.sep +"img").replace(".png", "_sol.npy"))

# 	# if abs(imgTime - datTime) > 1:
# 	# 	print(imgTime, datTime, imgTime - datTime)
# 	# 	# print(image.path, image.path.replace("img", "data"+ os.sep +"img").replace(".png", "_all.npy"))

# 	quantum.circuit.plot_results(counts, image.path, (topStat, minE))
# 	# reporting.visualize(reports, maxP, image.path)
	

