import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=['train']

catgies=[]

classes = []

out_file = open('lable_coordint.txt', 'w')
lable_file=open('catorgies.txt','w')
def convert_annotation(image_id):
    in_file = open('Gaothe_sample/Annotations/%s.xml'%(image_id))
    
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        #if cls not in classes or int(difficult)==1:
         #   continue
        #cls_id = classes.index(cls)
	if cls not in classes:
		classes.append(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        #bb = convert((w,h), b)
        #out_file.write(cls+" " + " ".join([str(a) for a in b]) + '\n')
	

wd = getcwd()

for image_set in sets:
    image_ids = open('Gaothe_sample/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    #list_file = open('%s_%s.txt'%(year, image_set), 'w')
    #classes = ["pedestrian","car","cyclist"]
    for image_id in image_ids:
     #   list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        convert_annotation(image_id)
    out_file.close()
    print(classes)
