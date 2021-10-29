import pickle
import os
from itertools import product
import numpy as np
def extract_expect_label_file_dict(original_dict,extract_tag,expect_label=['正常','加醋','涂碘']):
	extract_tag_file_list=[]
	for key,value in original_dict.items():
		if extract_tag in key:
			temp_label_list=[]
			for each_expect_label in expect_label:
				temp_label_list.append([])
			for file_name,label in value.items():
				if label in expect_label:
					temp_label_list[expect_label.index(label)].append(os.path.join(key,file_name))
			continue_flag=False
			temp_list_index_array=[]
			for each_list in temp_label_list:
				if len(each_list)==0:
					continue_flag=True
					break
				else:
					temp_list_index_array.append(np.arange(len(each_list)))
			if continue_flag:
				continue
			for each_index_list in product(*temp_list_index_array):
				temp_dict = {}
				valid_flag=True
				for i in range(len(each_index_list)):
					if not os.path.exists(temp_label_list[i][each_index_list[i]]):
						valid_flag=False
						break
					temp_dict[expect_label[i]]=temp_label_list[i][each_index_list[i]]
				if valid_flag:
					extract_tag_file_list.append(temp_dict)
	return extract_tag_file_list
# file_path="config_default.pth"
# with open(file_path,'rb') as f:
# 	data=pickle.load(f)
# file_data=data["file_path_label_dict"]
# tag="找不到"
# extract_tag_file_list=extract_expect_label_file_dict(file_data,tag)
# print("complete")

# def

