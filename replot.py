#!/usr/bin/env python3.9
# -*- coding:utf-8 -*-

# Use something like './replot.py storage/img/distribution/QAOA-MCP-medium-7.png' for only replotting a single image.

# Use './replot.py' for bulk replotting of all images (slow!).


import export.comparison
import export.distribution

from export.storage import folders


import numpy as np
import sys
import os

scan_only = sys.argv[1:]

folder: str = folders['comp'].replace("img", "data"+ os.sep +"img") # overview plot data
for imageData in os.scandir(folder):
	if imageData.is_dir():
		continue
	if "_all.npy" not in imageData.path:
		continue
	if scan_only and imageData.path.replace("data"+ os.sep +"img", "img").replace("_all.npy", ".png") not in scan_only:
		continue # faster testing
	try:
		reports, maxP, _ = np.load(imageData.path, allow_pickle=True) # _ = fileName
	except FileNotFoundError:
		continue
	# # print(folder + imageData, 'reports', maxP, fileName)
	# exceptions = ['allqAlgorithm-MCP-small-4.png', 'allqAlgorithm-MCP-tiny-4.png', 'VQEs-MCP-large-4.png']
	# assert fileName == imageData.path or imageData.name in exceptions, (fileName, folder, imageData.name)
	# imgTime = - 1645000000 + os.path.getmtime(imageData.path)
	# datTime = - 1645000000 + os.path.getmtime(imageData.path.replace("img", "data"+ os.sep +"img").replace(".png", "_all.npy"))

	# if imageData.name == 'allqAlgorithm-MCP-tiny-4.png':
	# 	reports = reversed(reports)
	# print(imgTime, datTime, imgTime - datTime)
	# # if abs(imgTime - datTime) > 1:
	# # 	print(imageData.path, imageData.path.replace("img", "data"+ os.sep +"img").replace(".png", "_all.npy"))
	# # print()

	varied = reports[0][2]
	# print(varied)
	export.comparison.plot(reports, maxP, imageData.path.replace("data"+ os.sep +"img", "img").replace("_all.npy", ".png"))
	print("Updated image: "+ imageData.path.replace("data"+ os.sep +"img", "img").replace("_all.npy", ".png"))
	


folder: str = folders['dist'] # histograms
for image in os.scandir(folder):
	if image.is_dir():
		continue
	if not (os.path.exists(image.path.replace("img", "data"+ os.sep +"img").replace(".png", ".npy")) and os.path.exists(image.path.replace("img", "data"+ os.sep +"img").replace(".png", "_sol.npy"))):
		# print(f"File {image.name.replace('.png', '_all.npy')} doesn't exist")
		continue
	if scan_only and image.path not in scan_only:
		continue # faster testing
	
	states, rates = np.load(image.path.replace("img", "data"+ os.sep +"img").replace(".png",     ".npy"), allow_pickle=True)
	topStat, minE = np.load(image.path.replace("img", "data"+ os.sep +"img").replace(".png", "_sol.npy"), allow_pickle=True)
	
	counts = {k:v for k,v in zip(states, rates)}


	# imgTime = - 1645000000 + os.path.getmtime(image.path)
	# datTime = - 1645000000 + os.path.getmtime(image.path.replace("img", "data"+ os.sep +"img").replace(".png", ".npy"))
	# datTime = - 1645000000 + os.path.getmtime(image.path.replace("img", "data"+ os.sep +"img").replace(".png", "_sol.npy"))

	# if abs(imgTime - datTime) > 1:
	# 	print(imgTime, datTime, imgTime - datTime)
	# 	# print(image.path, image.path.replace("img", "data"+ os.sep +"img").replace(".png", "_all.npy"))

	export.distribution.plot(counts, image.path, (topStat, minE))
	print("Updated image: "+ image.path)
	# reporting.visualize(reports, maxP, image.path)
	

