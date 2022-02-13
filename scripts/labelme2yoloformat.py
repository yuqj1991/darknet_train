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


annoImageDir = '../../dataset/roadSign/annoImage'
frameDir = '../../dataset/roadSign/images/'
labelsDir = '../../dataset/roadSign/labels'

classflyFile = "./roadSign_classfly_distance_data.txt"
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


# this api get the full image path from label txt file
def writeImageSet(srcDir, labelDir, setDir):
	if not os.path.exists(setDir):
		os.mkdir(setDir)
	dir_forder = ['train']
	for forder in dir_forder:
		set_file = setDir + '/' + forder + '.txt'
		set_file_ = open(set_file, 'w')
		labelfilelist = os.listdir(labelDir)
		for file in labelfilelist:
			img_file_path = os.path.abspath(srcDir + '/' + file.split('.txt')[0] + '.jpg') + '\n'
			set_file_.writelines(img_file_path)
			print(img_file_path)
		set_file_.close()


def shapes_to_label(jsonfilePath, label_name_to_value, root, classflydataFile):
	label_data = json.load(open(jsonfilePath, 'r'))
	imagePath = label_data['imagePath'].split('\\')[-1]
	fullPath = os.path.abspath(root + imagePath)
	print("fullPath: ", fullPath)
	img = cv2.imread(fullPath)
	img_h = img.shape[0]
	img_w = img.shape[1]
	classfly_file = open(classflydataFile, 'a+')
	label_shapes = label_data['shapes']
	for shape in label_shapes:
		label = shape['label']
		if label != 'Sidewalk a' and label != 'Sidewalk b' and label != 'side walk B' and label != 'side walk A' and label != 'Side walk B':
			print('imagepath: %s, jsonpath: %s'%(fullPath, jsonfilePath))
			assert imagePath.split('.jpg')[0] == jsonfilePath.split('/')[-1].split('.json')[0]
			labelfilePath = labelsDir + '/' + jsonfilePath.split('/')[-1].split('.json')[0] +'.txt'
			label_file_ = open(labelfilePath, 'a+')
			points = shape['points']
			xmin = points[0][0]
			ymin = points[0][1]
			xmax = points[1][0]
			ymax = points[1][1]
			if isSaveImglabeled:
				cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 3)
			b = (xmin, xmax, ymin, ymax)
			bb = convert((img_w, img_h), b)
			if collectBoxData:
				classfly_file.writelines(" ".join([str(a) for a in bb]) + '\n')
			#label_prefix = label.split(' ')
			#label_content = label_prefix[0] + ' ' + label_prefix[1]
			cls_id = label_name_to_value[label]
			label_file_.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
			label_file_.close()
	if isSaveImglabeled:
		if not os.path.exists(annoImageDir):
			os.mkdir(annoImageDir)
		labelImagePath = os.path.abspath(annoImageDir + '/' + imagePath)
		cv2.imwrite(labelImagePath, img)
	classfly_file.close()


def convert2labelFormat(jsonDir, labelDir, labelmapfile):
	classLabels = generatelabelSign(labelmapfile)
	classfy_ = open(classflyFile, "w")
	classfy_.truncate()
	classfy_.close()
	if not os.path.exists(labelDir):
		os.mkdir(labelsDir)
	n = 0
	if yoloformat:
		for json_file_ in os.listdir(jsonDir):
			json_path = os.path.join(jsonDir, json_file_)
			if os.path.isfile(json_path):
				shapes_to_label(json_path, classLabels, frameDir, classflyFile)
				n += 1
				print("num: ", n)
	else:
		raise Exception("please make sure yoloformat is true")

 
def main(args):
	jsonfileDir = args.jsonDir
	labelmapfile = args.labelmap
	convert2labelFormat(jsonfileDir, labelsDir, labelmapfile)
	setDir = "../../dataset/roadSign/Main"
	writeImageSet(frameDir, labelsDir, setDir)


if __name__ == '__main__':
	args = parse_args_augment()
	main(args)
