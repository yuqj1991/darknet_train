import os
import sys
import numpy as np
import cv2
import math
import argparse
import random

dataset_file = ['trainvalno5k.txt', '5k.txt']

dataset_new_file = ['trainperson.txt', 'val_yuqianjin.txt']

rootFolder = "/home/ubuntu/sharedata/coco/"
classflyFile = "/home/ubuntu/sharedata/coco/person_classfly_distance_data.txt"
boolClusterBbox = True


def showCropImageAddClassfly_data(FilterReplicateFolder, updatedNewFolder, newSetFile):
    if not os.path.exists(FilterReplicateFolder):
        raise("this folder does not exits!")
    setNewFile = open(rootFolder + newSetFile, "w")
    if boolClusterBbox:
        classfly_file = open(classflyFile, 'a+')
    for label_file in os.listdir(FilterReplicateFolder):
        file_name = label_file.replace(".txt", ".jpg")
        anno_file = FilterReplicateFolder + '/' + label_file
        img_file = rootFolder + 'images/train2014/' + file_name 
        if os.path.exists(anno_file) and os.path.exists(img_file):
            img = cv2.imread(img_file)
            h, w, _ = img.shape
            boxes = np.loadtxt(anno_file).reshape(-1, 5)
            if len(boxes) > 0:
                for _, bbox in enumerate(boxes):
                    xmin = int((bbox[1] - bbox[3] / 2) * w)
                    xmax = int((bbox[1] + bbox[3] / 2) * w)
                    ymin = int((bbox[2] - bbox[4] / 2) * h)
                    ymax = int((bbox[2] + bbox[4] / 2) * h)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255),thickness=3)
                    if boolClusterBbox:
                        if newSetFile != "5k.txt":
                            ################### cluster BBox ############################
                            ## get relative x, y , w, h corresponsed width, height#######
                            ################### cluster BBox ############################
                            classfly_content = str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + " " + str(bbox[4])+'\n'
                            classfly_file.writelines(classfly_content)
                            ###################### end cluster BBox ####################
            save_file = updatedNewFolder + "/" + file_name
            setNewFile.write(img_file + "\n")
            cv2.imwrite(save_file, img)
        else:
            raise("{} or {} is not existed!\n".format(anno_file, img_file))
    setNewFile.close()
    if boolClusterBbox: 
        classfly_file.close()


def compute_RGB_mean(img, boxes):
    h, w, _ = img.shape
    per_image_Bmean = []
    per_image_Gmean = []
    per_image_Rmean = []
    for _, box in enumerate(boxes):
        x1 = int((float(box[1]) - float(box[3])/2) * w)
        x2 = int((float(box[1]) + float(box[3])/2) * w)
        y1 = int((float(box[2]) - float(box[4])/2) * h)
        y2 = int((float(box[2]) + float(box[4])/2) * h)
        if y2 > y1 and x2 > x1:
            
            imghuman = img[y1:y2,x1:x2]
            per_image_Bmean.append(np.mean(imghuman[:,:,0]))
            per_image_Gmean.append(np.mean(imghuman[:,:,1]))
            per_image_Rmean.append(np.mean(imghuman[:,:,2]))
        else:
            print("invaild labels")
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    return R_mean, G_mean, B_mean


def generateRandomCropAnnoBBox(rawLabelFolder, rawImageFolder, mean_area = 0.0834):
    if not os.path.exists(rawLabelFolder):
        raise("{} is not exists".format(rawLabelFolder))
    if not os.path.exists(rawImageFolder):
        raise("{} is not exists".format(rawImageFolder))
    for label_file in os.listdir(rawLabelFolder):
        file_name = label_file.split(".txt")[0]
        img_file = rawImageFolder + '/' + file_name + ".jpg"
        anno_file = rawLabelFolder + '/' + label_file
        if os.path.exists(anno_file) and os.path.exists(img_file):
            boxes = np.loadtxt(anno_file).reshape(-1, 5)
            img = cv2.imread(img_file)
            R_mean, G_mean, B_mean = compute_RGB_mean(img, boxes)
            h, w, _ = img.shape
            new_anno_boxes = []
            for _, bbox in enumerate(boxes):
                single_area = bbox[3] * bbox[4]
                x1 = int((float(bbox[1]) - float(bbox[3])/2) * w)
                x2 = int((float(bbox[1]) + float(bbox[3])/2) * w)
                y1 = int((float(bbox[2]) - float(bbox[4])/2) * h)
                y2 = int((float(bbox[2]) + float(bbox[4])/2) * h)
                bbox_width = int(bbox[3] * w)
                bbox_height = int(bbox[4] * h)
                _, new_box_center_x, new_box_center_y, new_box_width, new_box_height = bbox
                if single_area >= mean_area:
                    y_left = 0.5 * y2 + 0.5 * y1
                    y_right = 0.6 * y2 + 0.4 * y1
                    y = random.randint(int(y_left),int(y_right) - 1) #取人区域高度的0.5~0.7的区域，随机选取起始x坐标
                    x_left1 = x1
                    x_right1 = x_left1 + int(0.3*bbox_width)
                    x_left2 = 0.5 * x2 + 0.5 * x1
                    x_right2 = 0.7 * x2 + 0.3 * x1
                    z = np.random.randint(2) #随机生成0或者1
                    if z==0: #取宽度的左边小区域
                        x = random.randint(int(x_left1),int(x_right1)-1)
                    else:
                        x = random.randint(int(x_left2),int(x_right2)-1)#取宽度的右边小区域
                    h_x = int(0.3 * bbox_width) #遮挡高度为人框的30%
                    h_y = int(0.3 * bbox_height) #遮挡宽度为人框的30%
                    img[y:y+h_y, x:x+h_x] = [int(B_mean), int(G_mean), int(R_mean)]

                    if (x+h_x) <= 0.6*x1 + 0.4*x2: #如果x2在0.2宽度左边，取右边的区域为新的区域
                        new_box_center_x = float(0.5*(x2+(x+h_x))/w)
                        new_box_width = float((x2-(x+h_x))/w)
                    elif x >= 0.6*x2 + 0.4*x1: #如果x2在0.2宽度右边，取左边的区域为新的区域
                        new_box_center_x = float(0.5*(x+x1)/w)
                        new_box_width = float((x-x1)/w)
                    else: #其他取y上面的区域
                        new_box_center_y = float(0.5*(y+y1)/h)
                        new_box_height = float((y-y1)/h)
                    new_anno_boxes.append("0 " + str(new_box_center_x) +" " + str(new_box_center_y) +" " + str(new_box_width) +" " + str(new_box_height))

                if len(new_anno_boxes) > 0:
                    crop_label = file_name + "_new.txt"
                    crop_label_file = open(rawLabelFolder + "/" + crop_label, "w")
                    crop_label_file.write('\n'.join(new_anno_boxes))
                    crop_label_file.close()
                    new_img_file = rawImageFolder + '/' + file_name + "_new.jpg"
                    cv2.imwrite(new_img_file, img)


def Boxenvelope(box, boxes, w, h):
    box_xmin = int((box[1] - float(box[3] / 2)) * w)
    box_xmax = int((box[1] + float(box[3] / 2)) * w)
    box_ymin = int((box[2] - float(box[4] / 2)) * h)
    box_ymax = int((box[2] + float(box[4] / 2)) * h)
    boolenvelope = False
    envelopeCount = 0
    for _, bbox in enumerate(boxes):
        if (box == bbox).all():
            continue
        else:
            xmin = int((bbox[1] - float(bbox[3] / 2)) * w)
            xmax = int((bbox[1] + float(bbox[3] / 2)) * w)
            ymin = int((bbox[2] - float(bbox[4] / 2)) * h)
            ymax = int((bbox[2] + float(bbox[4] / 2)) * h)
            if box_xmin <= xmin and box_xmax >= xmax and box_ymin <= ymin and box_ymax >= ymax:
                boolenvelope = True
                envelopeCount += 1
    return boolenvelope, envelopeCount
'''
to extract cleaned dataset which has big box annotation labels.
'''
def generate_replicate_label(rawLabelFolder, rawImageFolder, saveNewLabelFolder, 
                                    saveNewOriimgFolder, saveCutimgFolder, 
                                    smallThrehold = 0.5, bigThrehold = 0.85, boxNum = 7, proportion = 7.5):
    if not os.path.exists(rawLabelFolder):
        raise("{} is not exists".format(rawLabelFolder))
    if not os.path.exists(saveNewLabelFolder):
        os.mkdir(saveNewLabelFolder)
    if not os.path.exists(saveNewOriimgFolder):
        os.mkdir(saveNewOriimgFolder)
    if not os.path.exists(saveCutimgFolder):
        os.mkdir(saveCutimgFolder)
    count = 0
    for label_file in os.listdir(rawLabelFolder):
        file_name = label_file.split(".txt")[0]
        img_file = rawImageFolder + '/' + file_name + ".jpg"
        anno_file = rawLabelFolder + '/' + label_file
        if os.path.exists(anno_file) and os.path.exists(img_file):
            boolBigImage = False
            anno_bboxes = ''
            boolWritelabel = False
            boxes = np.loadtxt(anno_file).reshape(-1, 5)
            index = np.where(boxes[:,3] >= smallThrehold)[0] #if having big box anno label.
            if len(index)> 0 :
                boolBigImage = True
            else:
                continue
            img = cv2.imread(img_file)
            cut_img = cv2.imread(img_file)
            h, w, _ = img.shape
            if len(boxes) > 0:
                for _, bbox in enumerate(boxes):
                    xmin = int((bbox[1] - float(bbox[3] / 2)) * w)
                    xmax = int((bbox[1] + float(bbox[3] / 2)) * w)
                    ymin = int((bbox[2] - float(bbox[4] / 2)) * h)
                    ymax = int((bbox[2] + float(bbox[4] / 2)) * h)
                    bbox_width = xmax -xmin
                    bbox_height = ymax - ymin
                    boolNotSuperBig = True
                    boolNotBoxenvelope = True
                    if len(boxes) > boxNum and bbox[3] >= bigThrehold: # absolute super big big box anno
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255),thickness=3)
                        boolNotSuperBig = False
                        boolWritelabel = True
                    boolEnv, boxCount = Boxenvelope(bbox, boxes, w, h)
                    envBoxNum = 6
                    if float(bbox_width / bbox_height) >= proportion:
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255),thickness=3)
                        boolNotSuperBig = False
                        boolWritelabel = True
                    if len(boxes) > boxNum and boolEnv and boxCount >= envBoxNum: # related box envelope other small boxes
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255),thickness=3)
                        boolNotBoxenvelope = False
                        boolWritelabel = True
                    if boolBigImage and boolNotBoxenvelope and boolNotSuperBig:
                        label_content = "0 " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + " " + str(bbox[4])+"\n"
                        anno_bboxes += label_content
                        cv2.rectangle(cut_img, (xmin, ymin), (xmax, ymax), (0,0,255),thickness=3)
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255),thickness=3)
            if boolBigImage and anno_bboxes != '' and len(anno_bboxes) > 0 and boolWritelabel:
                New_label_path = saveNewLabelFolder + '/' + label_file
                New_label_file = open(New_label_path, "w")
                New_label_file.write(anno_bboxes)
                New_label_file.close()
                ori_img_file = saveNewOriimgFolder + '/' + file_name + ".jpg"
                after_image_file = saveCutimgFolder + "/" + file_name + ".jpg"
                cv2.imwrite(ori_img_file, img)
                cv2.imwrite(after_image_file, cut_img)
                count += 1
                print("count: {}, img_file: {}\n".format(count, ori_img_file))
        else:
            print("{} or {} is not existed!\n".format(anno_file, img_file))


def get_person_labelfile(setFile, newFile):
    if not os.path.exists(rootFolder + "labels/"):
        os.mkdir(rootFolder + "labels/")

    if not os.path.exists(rootFolder + "labels/train2014"):
        os.mkdir(rootFolder + "labels/train2014")

    if not os.path.exists(rootFolder + "labels/val2014"):
        os.mkdir(rootFolder + "labels/val2014")
    if setFile == "5k.txt":
        setNewFile = open(rootFolder + newFile, "w")

    with open(rootFolder + setFile, 'r') as gtfile:
         while True:
            img_path = gtfile.readline().strip()
            if img_path == "":
                break
            label_path = img_path.replace("images", "labels_originals").replace(".png", ".txt").replace(".jpg", ".txt")
            New_label_path = label_path.replace("labels_originals", "labels")
            if os.path.exists(label_path) and os.path.exists(img_path):
                boxes = np.loadtxt(label_path).reshape(-1, 5)
                boxes = [box for box in boxes if box[0] == 0]
                if len(boxes) > 0:
                    img = cv2.imread(img_path)
                    h, w, _ = img.shape
                    labels_list = []              
                    for _, bbox in enumerate(boxes):
                        box_width = bbox[3] * w
                        box_height = bbox[4] * h
                        if bbox[0] == 0 and box_height >= 32 and box_width >= 32:
                            label_content = "0 " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + " " + str(bbox[4])
                            labels_list.append(label_content)
                    if len(labels_list) > 0:
                        if setFile == "5k.txt":
                            setNewFile.write(img_path.replace("ubuntu/sharedata", "yuqianjin/temp") + "\n")
                        new_label_file = open(New_label_path, "w")
                        new_label_file.write('\n'.join(labels_list))
                        new_label_file.close()
    gtfile.close()
    if setFile == "5k.txt":
        setNewFile.close()

def parse_augement():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boolGeneratePersonLabel', action='store_true')
    parser.add_argument('--boolFilterSuperBox', action='store_true')
    parser.add_argument('--boolGenerateCropAnnoBBox', action='store_true')
    parser.add_argument('--boolShowRandomAnnoBox', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_augement()
    boolGeneratePersonLabel = args.boolGeneratePersonLabel
    if boolGeneratePersonLabel:
        if boolClusterBbox:
            classfy_ = open(classflyFile, "w")
            classfy_.truncate()
            classfy_.close()
        for index in range(2):
            get_person_labelfile(dataset_file[index], dataset_new_file[index])
    
    boolFilterSuperBox = args.boolFilterSuperBox
    if boolFilterSuperBox: ##filter super big box envelope more people
        rawLabelFolder = "/home/ubuntu/sharedata/coco/labels/train2014"
        rawImageFolder = "/home/ubuntu/sharedata/coco/images/train2014"
        saveNewLabelFolder = "/home/ubuntu/sharedata/coco/labels/train2014"
        saveNewOriimgFolder = "/home/ubuntu/sharedata/coco/images/replicate_2014_ori"
        saveCutimgFolder = "/home/ubuntu/sharedata/coco/images/replicate_2014_cut"
        generate_replicate_label(rawLabelFolder, rawImageFolder, saveNewLabelFolder, 
                                    saveNewOriimgFolder, saveCutimgFolder)

    boolGenerateCropAnnoBBox = args.boolGenerateCropAnnoBBox
    if boolGenerateCropAnnoBBox: ## generate random Crop image & label
        rawLabelFolder = "/home/ubuntu/sharedata/coco/labels/train2014"
        rawImageFolder = "/home/ubuntu/sharedata/coco/images/train2014"
        generateRandomCropAnnoBBox(rawLabelFolder, rawImageFolder)

    boolShowRandomAnnoBox = args.boolShowRandomAnnoBox
    if boolShowRandomAnnoBox:
        if boolClusterBbox:
            classfy_ = open(classflyFile, "w")
            classfy_.truncate()
            classfy_.close()
        label_set = "/home/ubuntu/sharedata/coco/labels/train2014"
        show_set = "/home/ubuntu/sharedata/coco/images/show_2014"
        showCropImageAddClassfly_data(label_set, show_set, dataset_new_file[0])

