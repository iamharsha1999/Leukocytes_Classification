import csv
import cv2
import os

base_path = '/home/harsha/Deep_Learning/Research Project/Dataset'
rows = []
for class_folder in os.listdir(base_path):
    if class_folder != 'TEST':
        for image in os.listdir(base_path + '/' + class_folder):
            row = []
            image_path = str(base_path + '/' + class_folder + '/' + image)
            row.append(image_path)
            row.append(str(class_folder))
            rows.append(row)

with open('dataset.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(rows)):
        filewriter.writerow(rows[i])
