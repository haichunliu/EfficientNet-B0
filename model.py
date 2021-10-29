from efficientnet_pytorch import EfficientNet
from torch import nn
import torch.nn.functional as F
import torch
from  torchvision import models
import copy

class sequential_image_classifier_efficientnet(nn.Module):
	def __init__(self,n_class):
		super(sequential_image_classifier_efficientnet, self).__init__()
		self.normal_feature_extractor=EfficientNet.from_pretrained("efficientnet-b0")
		self.vinegar_feature_extractor=EfficientNet.from_pretrained("efficientnet-b0")
		self.iodine_feature_extractor=EfficientNet.from_pretrained("efficientnet-b0")
		self.adpative_global_pooling=nn.AdaptiveAvgPool2d(output_size=1)
		self.gru=nn.GRU(1280,256,1,batch_first=True,bidirectional=True)
		self.fc=nn.Sequential(
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ELU(True),
			nn.Dropout(0.5),
			nn.Linear(256, 128),
			nn.BatchNorm1d(128),
			nn.ELU(True),
			nn.Dropout(0.5),
			nn.Linear(128, 64),
			nn.BatchNorm1d(64),
			nn.ELU(True),
			nn.Dropout(0.5),
			nn.Linear(64, 32),
			nn.BatchNorm1d(32),
			nn.ELU(True),
			nn.Dropout(0.5),
			nn.Linear(32, 16),
			nn.BatchNorm1d(16),
			nn.ELU(True),
			nn.Dropout(0.5),
			nn.Linear(16, 8),
			nn.BatchNorm1d(8),
			nn.ELU(True),
			nn.Dropout(0.5),
			nn.Linear(8, n_class)
		)
	def forward(self,normal_image,vinegar_image,iodine_image):
		normal_feature=self.normal_feature_extractor.extract_features(normal_image)
		normal_feature=self.adpative_global_pooling(normal_feature)
		vinegar_feature=self.vinegar_feature_extractor.extract_features(vinegar_image)
		vinegar_feature=self.adpative_global_pooling(vinegar_feature)
		iodine_feature=self.iodine_feature_extractor.extract_features(iodine_image)
		iodine_feature=self.adpative_global_pooling(iodine_feature)
		feature_sequeence=torch.cat((normal_feature,vinegar_feature,iodine_feature),dim=2)[:,:,:,0].permute((0, 2, 1))
		rnn_feature,_=self.gru(feature_sequeence)
		rnn_feature=rnn_feature[:,-1,:]
		out=self.fc(rnn_feature)
		return out


