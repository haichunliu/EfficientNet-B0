from dataset import Dataset_for_multimodule_img
from model import sequential_image_classifier_efficientnet
from util import *
import torch
from torch import nn,optim
from torchvision import transforms
from  sklearn.model_selection import train_test_split
import torch.utils.data as dataf
import pickle
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random
if __name__=="__main__":
	model_save_path="./save_model"
	if not os.path.exists(model_save_path):
		os.makedirs(model_save_path)
	config_file_path=r"阴性_HSIL_LSIL_info.pth"
	mean_std_file_path="ydj_mean_std_tensor_version_rgb.pth"
	# label_weight = torch.FloatTensor([1, 4])
	with open(config_file_path,'rb') as f:
		config_data=pickle.load(f)
	with open(mean_std_file_path,'rb') as f:
		mean_std_data=pickle.load(f)
	label_name=["阴性","LSIL","HSIL"]
	file_list=[]
	label_list=[]
	for i in range(len(label_name)):
		temp_data_list=extract_expect_label_file_dict(config_data[label_name[i]],label_name[i])
		if i !=0:
			temp_data_list=random.sample(temp_data_list,2500)
		file_list.append(temp_data_list)
		label_list.append([i]*len(temp_data_list))
	tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
		transforms.Normalize(mean_std_data["mean_tensor"], mean_std_data["std_tensor"]),])
	total_repeat=10
	for each_repeat in range(total_repeat):
		X_Train=[]
		X_Test=[]
		Y_Train=[]
		Y_Test=[]
		for i in range(len(label_name)):
			x_train, x_test, y_train, y_test = train_test_split(file_list[i], label_list[i], test_size=0.2)
			X_Train+=x_train
			X_Test+=x_test
			Y_Train+=y_train
			Y_Test+=y_test
		with open(os.path.join(model_save_path, "train_test_split_repeat_{0}.pth".format(each_repeat)), "wb") as f:
			pickle.dump({"X_train": X_Train, "X_test": X_Test, "Y_train": Y_Train, "Y_test": Y_Test}, f)
		train_dataset=Dataset_for_multimodule_img(X_Train,Y_Train,tfms)
		test_dataset=Dataset_for_multimodule_img(X_Test,Y_Test,tfms)
		learning_rate=0.1
		batch_size=32
		number_of_epoch=120
		save_epoch=10
		train_loader=dataf.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=3)
		test_loader=dataf.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=3)
		model=sequential_image_classifier_efficientnet(3)
		use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
		if use_gpu:
			device="cuda:0"
		else:
			device="cpu"
		model = model.to(device)
		# label_weight=label_weight.to(device)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(model.parameters(),momentum=0.9,lr=learning_rate,weight_decay=1e-5)
		scheduler = optim.lr_scheduler.StepLR(optimizer,10, 0.95)
		temp_test_acc=0.5
		for epoch in tqdm(range(number_of_epoch)):
			print('epoch {}'.format(epoch + 1))
			print('*' * 10)
			train_gt = []
			train_pred = []
			test_gt = []
			test_pred = []
			running_loss = 0.0
			running_acc = 0.0
			model.train()
			for i, data in enumerate(train_loader, 1):
				optimizer.zero_grad()
				normal_img, vinegar_img, iodine_img , label = data
				train_gt += label.tolist()
				if not isinstance(normal_img, torch.FloatTensor):
					normal_img = normal_img.type(torch.FloatTensor)
				normal_img = normal_img.to(device)
				if not isinstance(vinegar_img,torch.FloatTensor):
					vinegar_img=vinegar_img.type(torch.FloatTensor)
				vinegar_img = vinegar_img.to(device)
				if not isinstance(iodine_img, torch.FloatTensor):
					iodine_img=iodine_img.type(torch.FloatTensor)
				iodine_img=iodine_img.to(device)
				label = label.to(device)
				out = model(normal_img,vinegar_img,iodine_img)
				loss = criterion(out, label)
				running_loss += loss.item() * normal_img.size(0)
				_, pred = torch.max(out, 1)
				train_pred += pred.tolist()
				num_correct = (pred == label).sum()
				accuracy = (pred == label).float().mean()
				running_acc += num_correct.item()
				loss.backward()
				optimizer.step()
			print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
				epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))
			print("train_confusion_matrix:")
			print(confusion_matrix(train_gt, train_pred))
			model.eval()
			eval_loss = 0
			eval_acc = 0
			with torch.no_grad():
				for data in test_loader:
					normal_img, vinegar_img, iodine_img, label = data
					test_gt += label.tolist()
					if not isinstance(normal_img, torch.FloatTensor):
						normal_img = normal_img.type(torch.FloatTensor)
					normal_img = normal_img.to(device)
					if not isinstance(vinegar_img, torch.FloatTensor):
						vinegar_img = vinegar_img.type(torch.FloatTensor)
					vinegar_img = vinegar_img.to(device)
					if not isinstance(iodine_img, torch.FloatTensor):
						iodine_img = iodine_img.type(torch.FloatTensor)
					iodine_img = iodine_img.to(device)
					label = label.to(device)
					out = model(normal_img,vinegar_img,iodine_img)
					loss = criterion(out, label)
					eval_loss += loss.item() * normal_img.size(0)
					_, pred = torch.max(out, 1)
					test_pred += pred.tolist()
					num_correct = (pred == label).sum()
					eval_acc += num_correct.item()
				print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
					test_dataset)), eval_acc / (len(test_dataset))))
				print("test_confusion_matrix")
				print(confusion_matrix(test_gt, test_pred))
			if eval_acc / len(test_dataset) > temp_test_acc :
				torch.save(model.state_dict(),
						   os.path.join(model_save_path,r'efficientnet_classifier_HC_HSIL_{2}_{0}_{1}_{3}.pth'.format(epoch, eval_acc / (len(test_dataset)), each_repeat,
																running_acc / (len(train_dataset)))))
				temp_test_acc = eval_acc / len(test_dataset)
			if (epoch+1)%save_epoch==0:
				torch.save(model.state_dict(),
						   os.path.join(model_save_path,r'efficientnet_classifier_HC_HSIL_{2}_{0}_{1}_{3}.pth'.format(epoch,
										eval_acc / (len(test_dataset)),each_repeat,running_acc / (len(train_dataset)))))
			scheduler.step()
	print("complete!")