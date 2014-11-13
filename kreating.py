import os
import glob
import random


os.system("mkdir train")
os.system("mkdir test")

file = open('koching/trainpaths.txt', 'r')

for line in file:
	print line
	line = line.split("\n")
	line = line[0]
	lineFull = "/home/junior/Desktop/koching/" + line
	# print line.split("train/")
	lineMkDir = line.split("train/")[1]
	os.system("mkdir train/"+lineMkDir)
	os.system("mkdir test/"+lineMkDir)

	imgs = []
	for img in os.listdir(lineFull):
		if img.endswith(".tif"):
			imgs.append(img)
	
	imgs.sort()

	print len(imgs)
	# for img in imgs:
	# 	print line+"/"+img
	imgsSelected = []
	for i in range(0, 5):
		aux = ""
		img = random.choice(imgs)
		imgsSelected.append(img)
		imgs.remove(img)
		aux = line + "/" + img
	

	for img in imgs:	
		cmd = ""
		cmd = "cp " + lineFull + "/" + img + " /home/junior/Desktop/" + line + "/"
		print cmd
		os.system(cmd)

	for img in imgsSelected:
		line = line.split("train/")
		line = line[-1]
		cmd = ""
		cmd = "cp " + lineFull + "/" + img + " /home/junior/Desktop/test/" + line + "/"
		print cmd
		os.system(cmd)