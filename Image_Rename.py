import os
import cv2

filepath = 'Dataset'

for folder in os.listdir(filepath):
	if folder == 'EOSINOPHILS' or folder == 'MONOCYTES' or folder == 'NEUTROPHILS':
		print(folder)
		for image in os.listdir(filepath + '/' + folder):
			img  = cv2.imread(filepath + '/' + folder + '/' + image)
			img = cv2.resize(img , (620,620))
			cv2.imwrite(filepath + '/' + folder + '/' + image, img)    
