#!/usr/bin/python

import os
import sys
import zipfile
import csv
import codecs

if len(sys.argv) < 2:
	print("usage: strip_newlines.py INFILE")
	exit()

flag = False
content = ""

with zipfile.ZipFile(sys.argv[1]) as z:
	name_in = z.namelist()[0]
	with z.open(name_in) as file_in:
		reader = csv.reader(codecs.iterdecode(file_in, 'utf-8'))
		for line in reader:
			print(line)


'''
		name_in = os.path.splitext(name_in)
		name_out = name_in[0] + "_clean" + name_in[1]
		with open(name_out, 'w') as file_out:
			for line in file_in:
				print(line)
				if re.search(",[01],[01],[01],[01],[01],[01]", line):
					file_out.write(content)
					file_out.write(line)
					content = ""
				else:
					content += line.rstrip()
'''
	
