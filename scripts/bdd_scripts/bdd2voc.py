import argparse
import json
import os
import cv2
import random
from xml.dom.minidom import Document
__author__ = 'yuqj'
__copyright__ = 'Copyright (c) 2018, deepano'
__email__ = 'yuqj@deepano.com'
__license__ = 'DEEPANO'


anno_root = "../../../dataset/car_person_data/bdd100k/Annotations/"
label_train_root = "../../../dataset/car_person_data/bdd100k/labels/100k/train/"
label_val_root = "../../../dataset/car_person_data/bdd100k/labels/100k/val/"
anno_image = "../../../dataset/car_person_data/bdd100k/annoImage/"
src_image_train_root = "../../../dataset/car_person_data/bdd100k/JPEGImages/100k/train/"
src_image_val_root = "../../../dataset/car_person_data/bdd100k/JPEGImages/100k/val/"
src_image_test_root = "../../../dataset/car_person_data/bdd100k/JPEGImages/100k/test/"
label_json_file = ["../../../dataset/car_person_data/bdd100k/labels/100k/train/bdd100k_labels_images_train.json",
                   "../../../dataset/car_person_data/bdd100k/labels/100k/val/bdd100k_labels_images_val.json"]
label_txt_root = [label_train_root, label_val_root]
src_image_root = [src_image_train_root, src_image_val_root]
category_label = ['traffic light','traffic sign', 'person', 'rider', 'bicycle', 'bus', 'car', 'caravan', 'motorcycle', 'trailer',
                  'train', 'truck']
thread_hold = 40
clusterLabelFile = 'clusterlabelFile.txt'

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('label_path', type=str, help="this should be label json file")
    parse.add_argument('det_path', type=str, help="this should be label or annotation file path")
    args = parse.parse_args()
    return args


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def label2det(frames, src_image_, label_):
    boxes = list()
    classfy_file = open(clusterLabelFile, 'a+')
    for frame in frames:
        image_file = frame['name']
        label_file = label_ + image_file.split('.')[0]+'.txt'
        anno_xml_file = anno_root + frame['name'].split('.')[0]+'.xml'
        anno_image_file = anno_image + frame['name']
        src_image_file = src_image_ + frame['name']
        if 1:
            print("label_txt_file: ", label_file)
            print("anno_xml_file: ", anno_xml_file)
            print("anno_image_file: ", anno_image_file)
            print("src_image_file: ", src_image_file)
        srcImage = cv2.imread(src_image_file)
        label_w_file = open(label_file, 'w')
        # xml_file define
        doc = Document()
        annotation = doc.createElement('annotation')  # annotation element
        doc.appendChild(annotation)
        folder = doc.createElement('folder')
        folder_name = doc.createTextNode('wider_face')
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
        for label in frame['labels']:
            if 'box2d' not in label:
                continue
            category_index = 0
            if label['category'] not in category_label:
                continue
            else:
                for ii in range(len(category_label)):
                    if label['category'] == category_label[ii]:
                        category_index = ii
                        break
            xy = label['box2d']
            if xy['x1'] >= xy['x2'] or xy['y1'] >= xy['y2']:
                continue
            box = {'name': frame['name'],
                   'timestamp': frame['timestamp'],
                   'category': label['category'],
                   'bbox': [xy['x1'], xy['y1'], xy['x2'], xy['y2']],
                   'score': 1}
            category = label["category"]
            x1 = float(xy['x1'])
            y1 = float(xy['y1'])
            x2 = float(xy['x2'])
            y2 = float(xy['y2'])
            w = srcImage.shape[1]
            h = srcImage.shape[0]
            b = (x1, y1, x2, y2)
            bb = convert((w, h), b)
            # labels.txt
            label_content = str(category_index) + " " + " ".join([str(a) for a in bb]) + '\n'
            label_w_file.writelines(label_content)
            classfy_file.writelines(" ".join([str(a) for a in bb])+ '\n')
            # anno image
            cv2.rectangle(srcImage, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))
            cv2.putText(srcImage, category, (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            # anno_xml_file
            objects = doc.createElement('objects')
            annotation.appendChild(objects)
            object_name = doc.createElement('name')
            object_name.appendChild(doc.createTextNode(category))
            objects.appendChild(object_name)
            boundbox = doc.createElement('boundingbox')  # boundbox
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
            boxes.append(box)
        cv2.imwrite(anno_image_file, srcImage)
        label_w_file.close()
        #xml file
        xml_file = open(anno_xml_file, 'w')
        xml_file.write(doc.toprettyxml(indent=''))
        xml_file.close()
    classfy_file.close()
    return boxes


# this api get the full image path from label txt file
def write_train_val_set(src_image_root, label_file_root, trainset_root):
    # src_image_root: '/home/deepano/workspace/dataset/car_person_data/bdd100k/JPEGImages/100k/'
    # label_file_root: '/home/deepano/workspace/dataset/car_person_data/bdd100k/labels/'
    # trainset_root: '/home/deepano/workspace/dataset/car_person_data/bdd100k/ImageSets/Main/'
    dir_forder = ['train', 'val']
    for forder in dir_forder:
        label_folder = label_file_root + forder + '/'
        set_file = trainset_root + forder + '.txt'
        set_file_ = open(set_file, 'w')
        filelist = os.listdir(label_folder)
        for file in filelist:
            img_file_path = src_image_root + forder + '/' + file.split('.txt')[0] + '.jpg' + '\n'
            set_file_.writelines(img_file_path)
            print(img_file_path)
        set_file_.close()


def shuffle_file(filename):
    f = open(filename, 'r+')
    lines = f.readlines()
    random.shuffle(lines)
    f.seek(0)
    f.truncate()
    f.writelines(lines)
    f.close()


def convert_labels(label_json_path, src_img_, label_):
    frames = json.load(open(label_json_path, 'r'))
    det = label2det(frames, src_img_, label_)


def main():
    classfy_ = open(clusterLabelFile, "w")
    classfy_.truncate()
    classfy_.close()
    for ii in range(len(src_image_root)):
        convert_labels(label_json_file[ii], src_image_root[ii], label_txt_root[ii])
    write_train_val_set('/home/deepano/workspace/dataset/car_person_data/bdd100k/JPEGImages/100k/',
                       '/home/deepano/workspace/dataset/car_person_data/bdd100k/labels/100k/',
                       '/home/deepano/workspace/dataset/car_person_data/bdd100k/ImageSets/Main/')
    dir_forder = ['train', 'val']
    for dir in dir_forder:
       file = '/home/deepano/workspace/dataset/car_person_data/bdd100k/ImageSets/Main/' + dir + '.txt'
       shuffle_file(file)


if __name__ == '__main__':
    main()
