import sys
import cv2
import os
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face

def getFilePathList(dirPath):
    filePathList = []
    for root, dirs, files in os.walk(dirPath):
        for file in files:
            filePathList.append(os.path.join(root, file))
    return filePathList
#获取给定的文件夹路径下每个文件的路径

def check_jpg(file):
    _, ext = os.path.splitext(file)
    if ext == '.png':
        return  False
    if ext == '.jpg':
        return False
    else:
        return True
#检测格式是否符合要求



def face_detect(filename):
    img = cv2.imread(filename)
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # b, g, r = cv2.split(img)
    # img2 = cv2.merge([r, g, b])

    bboxs, landmarks = mtcnn_detector.detect_face(img)
    filesavename = os.path.basename(filename)
    save_name = 'res_' + filesavename
    print(save_name)
    print(bboxs)

    print(landmarks)
    # print box_align

    vis_face(img_bg, bboxs, landmarks, save_name,  filename)

if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt", r_model_path="./original_model/rnet_epoch.pt", o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    dirPath = input("请输入一个文件夹路径：")
    filePathList = getFilePathList(dirPath)
    for filePath in filePathList:
        if  check_jpg(filePath):
            print(filePath)
            sys.exit("This is not a .jpg file. Please remove it from the folder.")

        filePathc =  filePath.replace('\\','/')
        print(filePathc)
        face_detect(filePathc)



