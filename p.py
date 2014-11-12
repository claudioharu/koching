import os
import glob

path = os.getcwd() + "/train" 
roots = []

for root, dirs, files in os.walk(path):
	roots.append(root)

roots = roots[1:]
roots.sort()

for root in roots:
	root = root.split("/home/junior/Desktop/koching/")[1:]
	print root[0]