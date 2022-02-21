# -*- coding: utf-8 -*-
import numpy as np
import json
import os, math
import cv2, random

data_desc = "2022_02_16"

annoJsonDir = '../jsonDir/' + data_desc
ocrDataSet = "../ocr_dataSet"
DataSetDir = '/scanImages/new_images'


verifyAnnoImages = True

classflyFile = "./roadSign_classfly_distance_data.txt"
YoloFormat = True

verifyDataSet = "../verifyDataset/" + data_desc

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
	return (x, y, w, h)

def convert_polygon(size, poly):
	dw = 1./(size[0])
	dh = 1./(size[1])
	new_poly = []
	for i in range(len(poly)):
		x, y = poly[i]
		x = x*dw
		y = y*dh
		new_poly.append(x)
		new_poly.append(y)
	return new_poly

annoLabelRoots = {
  "polyTask" : "../annoDir/page_label",
  "homeworkTask": "../annoDir/review_label",
  "ocrTask": "../annoDir/ocr_label"
}

ocr_labels = {
    "0" : 0,
    "1" : 1,
    "2" : 2,
    "3" : 3,
    "4" : 4,
    "5" : 5,
    "6" : 6,
    "7" : 7,
    "8" : 8,
    "9" : 9
}


polyGon_labels = {"left_page": 0,
                  "right_page": 1}

detection_labels = {
    "correct_sign": 0,
    "wrong_sign": 1,
    "page_number": 2,
    "Review_A": 3,
    "Review_B": 4,
    "Review_C": 5,
    "Review_D": 6,
    "Review_E": 7,
    "work_duration": 8,
}

color_labels = {
    "correct_sign":  (98, 9, 11),
    "wrong_sign": (38, 9, 11),
    "page_number": (98, 49, 11),
    "Review_A": (98, 29, 51),
    "Review_B": (37, 25, 61),
    "Review_C": (124, 19, 51),
    "Review_D": (23, 35, 61),
    "Review_E": (54, 76, 51),
    "work_duration": (25, 19, 41),
	"left_page": (220, 119, 141),
    "right_page": (120, 156, 41)
}

def parse_anno_json_file(anno_file, image):
    img_h, img_w = image.shape[:2]
    label_data = json.load(open(anno_file, 'r'))
    json_file = anno_file.split('/')[-1].strip("\n")
    ## 解析 多边形标注 (左页或者右页)
    label_shapes = label_data['shapes']
    poly_label_file = open(annoLabelRoots["polyTask"] + "/" +json_file.replace(".json", ".txt"), 'w')
    review_label_file = open(annoLabelRoots["homeworkTask"] + "/" +json_file.replace(".json", ".txt"), 'w')
    review_annoDesc = []
    polyPage_annoDesc = []
    ocr_count = 0
    for anno in label_shapes:
        label = anno["label"]
        verify_label = label
        if label in detection_labels.keys():
            label_id = detection_labels[label]
            leftTop_point = anno["left_up_points"]
            rightBottom_point = anno["right_bottom_points"]
            xmin = leftTop_point[0]
            xmax = rightBottom_point[0]
            ymin = leftTop_point[1]
            ymax = rightBottom_point[1]
            b = (xmin, xmax, ymax, ymin)
            bb = convert((img_w, img_h), b)
            annoDesc = str(label_id) + " " + " ".join([str(a) for a in bb]) + '\n'
            review_annoDesc.append(annoDesc)
            if label == "work_duration" or label == "page_number":
                x1 = xmin
                x2 = xmax
                y1 = ymax
                y2 = ymin
                #print("x1, ", x1, ", y1: ", y1, ", x2: ", x2, ", y2: ", y2)
                ocr_image = image[y1:y2, x1:x2, :]
                labelFile_name = str(ocr_count) + "_" +json_file.replace(".json", ".txt")
                ocr_image_name = ocrDataSet + "/" + str(ocr_count) + "_" +json_file.replace(".json", ".jpg")
                ocr_anno_file = open(
                    annoLabelRoots["ocrTask"] + "/" + labelFile_name, 'w')
                ocr = anno["ocr"]
                ocr_label = []
                for i in range(len(ocr)):
                    ocr_label.append(int(ocr[i]))
                #print("ocr_label: ", labelFile_name, ". ,", ocr_label)
                ocr_anno_file.writelines(" ".join([str(a) for a in ocr_label]) + '\n')
                ocr_anno_file.close()
                cv2.imwrite(ocr_image_name, ocr_image)
                ocr_count += 1
                verify_label = label + "_"+ ocr
            if verifyAnnoImages:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_labels[label], 2)
                cv2.putText(image, verify_label, (xmin, ymax), cv2.FONT_HERSHEY_PLAIN, 1, color_labels[label], 1)
        elif label in polyGon_labels.keys():
            poly_label_id = polyGon_labels[label]
            points = anno["points"]
            if verifyAnnoImages:
                cv2.polylines(image, [np.array(points)], True, color_labels[label], 2)
                cv2.putText(image, verify_label, (points[1][0], points[1][1]), cv2.FONT_HERSHEY_PLAIN, 1, color_labels[label], 1)
            points = convert_polygon((img_w, img_h), points)
            poly_annoDesc = str(poly_label_id) + " " + " ".join([str(b) for b in points]) + "\n"
            #print("poly_label: ", label, ". ,", poly_annoDesc)
            polyPage_annoDesc.append(poly_annoDesc)
        else:
            raise ValueError("there is no other label for homeAnno_task", label)
    poly_label_file.writelines(polyPage_annoDesc)
    review_label_file.writelines(review_annoDesc)
    poly_label_file.close()
    review_label_file.close()
    if verifyAnnoImages:
        cv2.imwrite(verifyDataSet + "/" + json_file.replace(".json", ".jpg"), image)

# this api get the full image path from label txt file
def writeImageSet(srcDir, labelDir, setDir, task):
    if not os.path.exists(setDir):
        os.mkdir(setDir)
    train_file_ = setDir + '/' + task + "_trainval.txt"
    val_file_ = setDir + '/' + task + "_val.txt"
    img_file_content = []
    labelfilelist = os.listdir(labelDir)
    for file in labelfilelist:
        img_file_path = os.path.abspath(srcDir + '/' + file.split('.txt')[0] + '.jpg') + '\n'
        img_file_content.append(img_file_path)
        print(img_file_path)
    random.shuffle(img_file_content)
    total_instances = len(img_file_content)
    trainval_num = math.ceil(total_instances * 0.8)
    val_files = open(val_file_, "w")
    train_files = open(train_file_, "w")
    train_files.writelines(img_file_content[0:trainval_num])
    train_files.close()
    val_files.writelines(img_file_content[trainval_num:])
    val_files.close()

def convert2labelFormat():
	# classLabels = ""
	# classfy_ = open(classflyFile, "w")
	# classfy_.truncate()
	# classfy_.close()
	n = 0
	if YoloFormat:
		for json_file_ in os.listdir(annoJsonDir):
			json_path = os.path.join(annoJsonDir, json_file_)
			if os.path.isfile(json_path):
				img_path = os.path.join(DataSetDir, json_file_.replace(".json", ".jpg"))
				print("img_path", img_path)
				src_img = cv2.imread(img_path)
				parse_anno_json_file(json_path, src_img)
				n += 1
				print("num: ", n)
	else:
		raise Exception("please make sure YoloFormat is true")


def main():
	convert2labelFormat()
	

if __name__ == '__main__':
	#main()
    setDir = "../annoDir/ImageSet"
    writeImageSet(DataSetDir, annoLabelRoots["polyTask"], setDir, "page")
    writeImageSet(DataSetDir, annoLabelRoots["homeworkTask"], setDir, "review")
    writeImageSet(ocrDataSet, annoLabelRoots["ocrTask"], setDir, "ocr")
