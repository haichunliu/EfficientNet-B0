import numpy as np
import mne
from PIL import Image
import torch.utils.data as dataf
import math
import matplotlib.pyplot as plt

class Dataset_for_multimodule_img(dataf.Dataset):
	def __init__(self,file_list,data_label_list,transform_model):
		self.file_list=file_list
		self.label_list=data_label_list
		self.transform_model=transform_model
		self.data_len=len(file_list)

	def __getitem__(self, index):
		file_path_dict=self.file_list[index]
		label=self.label_list[index]
		normal_image=self.transform_model(Image.open(file_path_dict["正常"]))
		vinegar_image=self.transform_model(Image.open(file_path_dict["加醋"]))
		iodine_image=self.transform_model(Image.open(file_path_dict["涂碘"]))
		return normal_image,vinegar_image,iodine_image,label

	def __len__(self):
		return self.data_len


