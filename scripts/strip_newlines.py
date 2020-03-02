#!/usr/bin/python

import os
import sys

if len(sys.argv) < 2:
	print("usage: strip_newlines.py INFILE")
	exit()

flag = False

with open(sys.argv[1]) as file_in:
	name_in = os.path.splitext(sys.argv[1])
	name_out = name_in[0] + "_clean" + name_in[1]
	with open(name_out, 'w') as file_out:
		for line in file_in:
			if line.rstrip().endswith("\"\"\""):
				flag = True
				file_out.write(line.rstrip())
				file_out.write("\t")
			elif line.startswith("\"\"\""):
				flag = False
				file_out.write(line.rstrip())
				file_out.write("\n")
			elif flag:
				file_out.write(line.rstrip())
				file_out.write("\t")
			else:
				file_out.write(line.rstrip())
				file_out.write("\n")

	
