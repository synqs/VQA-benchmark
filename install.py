#!/usr/bin/env python3.9
# -*- coding:utf-8 -*-

from export.storage import folders


import os
import stat


# Create folders
for folder in folders.values():
	os.makedirs(folder, mode=511, exist_ok=True)

additional_data_folders = ['comp', 'dist']
for additional_data_folder in additional_data_folders:
	os.makedirs(folders[additional_data_folder].replace("img", "data"+ os.sep +"img"), exist_ok=True)


# Grant permissions
executable_files = [
	'install.py',
	'main.py',
	'replot.py',
]
for ef in executable_files:
	mode = os.stat(ef).st_mode
	os.chmod(ef, mode | stat.S_IXUSR)


