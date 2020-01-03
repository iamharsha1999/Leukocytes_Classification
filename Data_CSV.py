import csv
import os
import xml.etree.ElementTree as ET

base_path = 'Dataset/Image_Classification'
rows = []

for folder in os.listdir(base_path):
    if str(folder) in ['Basophils', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Neutrophils']:
        for image in os.listdir(base_path + '/' + folder):
            row = []
            print(str(image))
            path_of_image = base_path + '/' + folder + '/' + image
            class_label = str(folder)
            row.append(path_of_image)
            row.append(class_label)
            rows.append(row)
    

with open('dataset.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(rows)):
        filewriter.writerow(rows[i])
