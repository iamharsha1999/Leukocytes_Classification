class FasterRCNN:

	def getdata(input_file_path):

		
		"""
			Input File Path is given as an input parameter

			Function returns all_data: list(filepath, width, height, list(bboxes))
							 classes_count: dict{key:class_name, value:count_num} 
							 e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
							 class_mapping: dict{key:class_name, value: idx}
							 e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
		"""

		found_bg = False
		all_imgs = {}
		classes_count = {}
		class_mapping = {}
		visualise = True

		i = 1

		file =  open(input_file_path, 'r')
		print("Parsing Annotation Files")

		for line in file:

			(file_name, x1, y1, x2, y2, class_name) = line.strip().split(',')

		if class_name not in classes_count:
				classes_count[class_name] = 1
		else:
			classes_count[class_name] += 1

		if class_name not in class_mapping:
			if class_name == 'bg' and found_bg == False:
				found_bg = True
			class_mapping[class_name] = len(class_mapping)
		
		if filename not in all_imgs:
				all_imgs[filename] = {}
				
				img = cv2.imread(filename)
				(rows,cols) = img.shape[:2]
				all_imgs[filename]['filepath'] = filename
				all_imgs[filename]['width'] = cols
				all_imgs[filename]['height'] = rows
				all_imgs[filename]['bboxes'] = []
				
		all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

		all_data = []
		for key in all_imgs:
			all_data.append(all_imgs[key])
		if found_bg:
			if class_mapping['bg'] != len(class_mapping) - 1:
				key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
				val_to_switch = class_mapping['bg']
				class_mapping['bg'] = len(class_mapping) - 1
				class_mapping[key_to_switch] = val_to_switch
		
		return all_data, classes_count, class_mapping
