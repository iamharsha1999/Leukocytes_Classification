import cv2
import os

data_path = 'Dataset/Image_Classification'

for folder in os.listdir(data_path):
    if str(folder) in ['Basophils', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Neutrophils']:
        a = 1
        for image in os.listdir(data_path + '/' + folder):
            img =  cv2.imread(data_path + '/' + folder + '/' + image)
            img_r_90_c = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img_r_90_c_2 = cv2.rotate(img_r_90_c, cv2.ROTATE_90_CLOCKWISE)
            img_r_90_cc = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_r_90_cc_2 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_f_ud = cv2.flip(img, 0)
            img_f_lr = cv2.flip(img, 1)
            img_f_udlr = cv2.flip(img, -1)
            cv2.imwrite(data_path + '/' + folder + '/a' + str(a) + '.jpg', img_r_90_c )
            a+=1
            cv2.imwrite(data_path + '/' + folder + '/a' + str(a) + '.jpg', img_r_90_c_2 )
            a+=1
            cv2.imwrite(data_path + '/' + folder + '/a' + str(a) + '.jpg', img_r_90_cc )
            a+=1
            cv2.imwrite(data_path + '/' + folder + '/a' + str(a) + '.jpg', img_r_90_cc_2 )
            a+=1
            cv2.imwrite(data_path + '/' + folder + '/a' + str(a) + '.jpg', img_f_ud )
            a+=1
            cv2.imwrite(data_path + '/' + folder + '/a' + str(a) + '.jpg', img_f_lr )
            a+=1
            cv2.imwrite(data_path + '/' + folder + '/a' + str(a) + '.jpg', img_f_udlr )
        print(str(folder) + "over")
