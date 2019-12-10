import cv2
import os
import xml.etree.ElementTree as ET

annot = '/home/harsha/Deep_Learning/Research/Dataset/SSD-7-DATA/Annotations'
dest = '/home/harsha/Deep_Learning/Research/Dataset/Image_Classification/Cropped_Images/'

for xml in os.listdir(annot):
    tree = ET.parse(annot + '/'  + xml)
    root = tree.getroot()
    print("XML Read")
    img_path = root[2].text
    img_name = (root[1].text)[:-4]
    x1 = int(root[6][4][0].text)
    y1 = int(root[6][4][1].text)
    x2 = int(root[6][4][2].text)
    y2 = int(root[6][4][3].text)
    img = cv2.imread(img_path)
    img = img[y1:y2,x1:x2,:]
    img = cv2.resize(img,(300,300))
    cv2.imwrite(dest + img_name + '.jpg', img)
    print("Image .{} Written".format(img_name + '.jpg'))
