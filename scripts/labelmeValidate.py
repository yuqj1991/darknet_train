# -*- coding: utf-8 -*-
import argparse
import json
import os
import cv2


def parse_args_augment():
	parser = argparse.ArgumentParser()
	parser.add_argument('jsonDir')
	parser.add_argument('labelmap')
	args = parser.parse_args()
	return args



rootDir = '../../dataset/roadSign'

wrongLabeledfile_ = 'wrongfile'

collectBoxData = True

isSaveImglabeled = True
yoloformat = True


def generatelabelSign(labelfile):
	classLabels = {}
	i = 0
	file_ = open(labelfile, 'r')
	for line in file_.readlines():
		curLine=line.strip().split('\n')
		classLabels[curLine[0]] = i
		i+=1
	return classLabels


def convert(size, box):
	dw = 1./(size[0])
	dh = 1./(size[1])
	x = (box[0] + box[1])/2.0 - 1
	y = (box[2] + box[3])/2.0 - 1
	w = box[1] - box[0]
	h = box[3] - box[2]
	x = x*dw
	w = w*dw
	y = y*dh
	h = h*dh
	return (x,y,w,h)


def shapes_to_label(jsonfilePath, label_name_to_value, root):
	label_data = json.load(open(jsonfilePath, 'r'))
	imagePath = label_data['imagePath'].split('..\\')[-1]
	print('image: %s, json: %s'%(imagePath.split('.jpg')[0], jsonfilePath.split('/')[-1].split('.json')[0]))
	#assert imagePath.split('.jpg')[0] == jsonfilePath.split('/')[-1].split('.json')[0]
	fullPath = os.path.abspath(root + '/images/' + imagePath)
	classfly_file = open(wrongLabeledfile_, 'a+')
	print(fullPath)
	img = cv2.imread(fullPath)
	img_h = img.shape[0]
	img_w = img.shape[1]
	label_shapes = label_data['shapes']
	for shape in label_shapes:
		label = shape['label']
		if label != 'Sidewalk a' and label != 'Sidewalk b' and label != 'side walk B' and label != 'side walk A' and label != 'Side walk B':
			if label not in label_name_to_value:
				classfly_file.write("jsonfile: " + jsonfilePath + ', wrong label: ' + label + '\n')
			if imagePath.split('.jpg')[0] != jsonfilePath.split('/')[-1].split('.json')[0]:
				classfly_file.write("jsonfile: " + jsonfilePath.split('/')[-1].split('.json')[0] + ',imagefile: ' + imagePath.split('.jpg')[0] + '\n')
	classfly_file.close()

def convert2labelFormat(jsonDir, labelmapfile):
	classLabels = generatelabelSign(labelmapfile)
	n = 0
	if yoloformat:
		for json_file_ in os.listdir(jsonDir):
			json_path = os.path.join(jsonDir, json_file_)
			if os.path.isfile(json_path):
				shapes_to_label(json_path, classLabels, rootDir)
			n += 1
			print('num: ', n)
	else:
		raise Exception("please make sure yoloformat is true")

 
def main(args):
	jsonfileDir = args.jsonDir
	labelmapfile = args.labelmap
	convert2labelFormat(jsonfileDir, labelmapfile)


if __name__ == '__main__':
	args = parse_args_augment()
	main(args)
