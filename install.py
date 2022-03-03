#!/usr/bin/env python3.9
# -*- coding:utf-8 -*-

from export.storage import folders


import os


for folder in folders.values():
	os.makedirs(folder, mode=511, exist_ok=True)

additional_data_folders = ['comp', 'dist']
for additional_data_folder in additional_data_folders:
	os.makedirs(folders[additional_data_folder].replace("img", "data"+ os.sep +"img"), exist_ok=True)



