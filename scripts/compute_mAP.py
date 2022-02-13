from .voc_eval import voc_eval
import numpy as np

def label2det(rawLabelFolder, rawImageFolder, anno_root):
	if not os.path.exists(rawLabelFolder):
		raise("{} is not exists".format(rawLabelFolder))
	if not os.path.exists(rawImageFolder):
		raise("{} is not exists".format(anno_root))
	if not os.path.exists(anno_root):
		os.mkdir(saveCutimgFolder)
	for label_file in os.listdir(rawLabelFolder):
		file_name = label_file.split(".txt")[0]
		image_file = rawImageFolder + '/' + file_name + ".jpg"
		label_file = rawLabelFolder + '/' + label_file
		anno_xml_file = anno_root + '/' + file_name +".xml"
		if os.path.exists(label_file) and os.path.exists(image_file):
			boxes = np.loadtxt(label_file).reshape(-1, 5)
			if len(boxes) > 0:
				srcImage = cv2.imread(image_file)
				# xml_file define
				doc = Document()
				annotation = doc.createElement('annotation')  # annotation element
				doc.appendChild(annotation)
				folder = doc.createElement('folder')
				folder_name = doc.createTextNode('coco_person')
				folder.appendChild(folder_name)
				annotation.appendChild(folder)
				filename_node = doc.createElement('filename')
				filename_name = doc.createTextNode(src_image_file)
				filename_node.appendChild(filename_name)
				annotation.appendChild(filename_node)
				source = doc.createElement('source')  # source sub_element
				annotation.appendChild(source)
				database = doc.createElement('database')
				database.appendChild(doc.createTextNode('bbd Database'))
				annotation.appendChild(database)
				annotation_s = doc.createElement('annotation')
				annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
				source.appendChild(annotation_s)
				image = doc.createElement('image')
				image.appendChild(doc.createTextNode('flick'))
				source.appendChild(image)
				flickrid = doc.createElement('flickid')
				flickrid.appendChild(doc.createTextNode('-1'))
				source.appendChild(flickrid)
				owner = doc.createElement('owner')  # company element
				annotation.appendChild(owner)
				flickrid_o = doc.createElement('flickid')
				flickrid_o.appendChild(doc.createTextNode('deepano'))
				owner.appendChild(flickrid_o)
				name_o = doc.createElement('name')
				name_o.appendChild(doc.createTextNode('deepano'))
				owner.appendChild(name_o)
				size = doc.createElement('size')  # img size info element
				annotation.appendChild(size)
				width = doc.createElement('width')
				width.appendChild(doc.createTextNode(str(srcImage.shape[1])))
				height = doc.createElement('height')
				height.appendChild(doc.createTextNode(str(srcImage.shape[0])))
				depth = doc.createElement('depth')
				depth.appendChild(doc.createTextNode(str(srcImage.shape[2])))
				size.appendChild(width)
				size.appendChild(height)
				size.appendChild(depth)
				for box in boxes:
					x1 = float(box[1] - float(box[3] / 2))
					y1 = float(box[2] - float(box[4] / 2))
					x2 = float(box[1] + float(box[3] / 2))
					y2 = float(box[2] + float(box[4] / 2))
					category = box[0]
					# anno_xml_file
					objects = doc.createElement('objects')
					annotation.appendChild(objects)
					object_name = doc.createElement('name')
					object_name.appendChild(doc.createTextNode(category))
					objects.appendChild(object_name)
					boundbox = doc.createElement('bndbox')  # boundbox
					objects.appendChild(boundbox)
					xmin = doc.createElement('xmin')
					xmin.appendChild(doc.createTextNode(str(x1)))
					boundbox.appendChild(xmin)
					ymin = doc.createElement('ymin')
					ymin.appendChild(doc.createTextNode(str(y1)))
					boundbox.appendChild(ymin)
					xmax = doc.createElement('xmax')
					xmax.appendChild(doc.createTextNode(str(x2)))
					boundbox.appendChild(xmax)
					ymax = doc.createElement('ymax')
					ymax.appendChild(doc.createTextNode(str(y2)))
					boundbox.appendChild(ymax)
				#xml file
				xml_file = open(anno_xml_file, 'w')
				xml_file.write(doc.toprettyxml(indent=''))
				xml_file.close()
rawLabelFolder = "/home/ubuntu/sharedata/coco/labels/val2014"
rawImageFolder = "/home/ubuntu/sharedata/coco/images/val2014"
anno_root = "/home/ubuntu/sharedata/coco/labels/val_xml2014"
label2det(rawLabelFolder, rawImageFolder, anno_root)
 
rec,prec,ap=voc_eval('../results/{}.txt', '/home/ubuntu/sharedata/coco/labels/val_xml2014/{}.xml', 
                            '/home/ubuntu/sharedata/coco/val.txt', 'person', '.')
 
print('rec',rec)
print('prec',prec)
print('ap',ap)
