import os
import torch
import numpy as np
from .constants import *
from .models import *

def convert_to_windows(data, model):
	data = torch.tensor(data).double()
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w)
	return torch.stack(windows)

def form_test_dataset(data):
	anomaly_per_dim = data > np.percentile(data, PERCENTILES, axis=0)
	anomaly_which_dim, anomaly_any_dim = [], []
	for i in range(0, data.shape[1], 3):
		anomaly_which_dim.append(np.argmax(data[:, i:i+3] + 0, axis=1))
		anomaly_any_dim.append(np.logical_or.reduce(anomaly_per_dim[:, i:i+3], axis=1))
	anomaly_any_dim = np.stack(anomaly_any_dim, axis=1)
	anomaly_which_dim = np.stack(anomaly_which_dim, axis=1)
	return anomaly_any_dim + 0, anomaly_which_dim

def load_npyfile(folder, fname):
	path = os.path.join(folder, fname)
	if not os.path.exists(path):
		raise Exception('Data not found ' + path)
	return np.load(path)

def load_dataset(folder, model):# 离线训练FPE时用到
	time_data = load_npyfile(folder, data_filename)
	time_data = normalize_time_data(time_data) # Normalize data
	train_schedule_data = torch.tensor(load_npyfile(folder, schedule_filename)).double()
	train_time_data = convert_to_windows(time_data, model)
	anomaly_data, class_data = form_test_dataset(time_data)
	return train_time_data, train_schedule_data, anomaly_data, class_data

def load_on_the_fly_dataset(model, folder, stats):# 在线微调FPE时调用
	train_time_data = load_npyfile(folder, data_filename)
	time_data = stats.time_series[-LATEST_WINDOW_SIZE:]
	time_data = normalize_test_time_data(time_data, train_time_data)
	train_schedule_data = stats.schedule_series[-LATEST_WINDOW_SIZE:]
	train_time_data = convert_to_windows(time_data, model)
	anomaly_data, class_data = form_test_dataset(time_data)
	return train_time_data, train_schedule_data, anomaly_data, class_data

def save_model(folder, fname, model, optimizer, epoch, accuracy_list):
	path = os.path.join(folder, fname)
	# if 'Att' in model.name: print(model.prototype)
	if 'G' in model.name or 'D' in model.name: model.prototype = {}
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_prototypes': model.prototype,
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list}, path)

def load_model(folder, fname, modelname):
	import recovery.PreGANSrc.src.models
	path = os.path.join(folder, fname)
	model_class = getattr(recovery.PreGANSrc.src.models, modelname)
	model = model_class().double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	if os.path.exists(path):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(path)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.prototype = checkpoint['model_prototypes']
		for p in model.prototype: p.requires_grad = False
		# if 'Att' in model.name: print(model.prototype)
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:# If model does not exist, create a new one
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, epoch, accuracy_list

def load_gan(folder, gfname, dfname, gmodelname, dmodelname):
	gmodel, gopt, epoch, accuracy_list = load_model(folder, gfname, gmodelname)
	dmodel, dopt, _, _ = load_model(folder, dfname, dmodelname)
	return gmodel, dmodel, gopt, dopt, epoch, accuracy_list

def save_gan(folder, gfname, dfname, gmodel, dmodel, gopt, dopt, epoch, accuracy_list):
	save_model(folder, gfname, gmodel, gopt, epoch, accuracy_list)
	save_model(folder, dfname, dmodel, dopt, 0, [])

# Misc
def normalize_time_data(time_data):
	return time_data / (np.max(time_data, axis = 0) + 1e-8) 

def normalize_test_time_data(time_data, train_time_data):
	return (time_data / (np.max(train_time_data, axis = 0) + 1e-8))

def run_simulation(stats, schedule_data):
    e, r = stats.runSimulation(schedule_data)
    score = Coeff_Energy * e + Coeff_Latency * r
    return score

def get_classes(embeddings, model):# 根据输入的嵌入向量（embeddings）和模型（model）的原型向量，确定每个嵌入向量所属的类别
	class_list = []
	for e in embeddings:# 遍历所有嵌入向量
		if (e == 0).all().item():# 如果嵌入向量全为0，则将其类别标记为-1，表示该嵌入向量不属于任何类别
			class_list.append(-1); continue
		distances = np.array([(torch.mean((e - p)**2)).item() for p in model.prototype])# 计算嵌入向量与每个原型向量之间的欧氏距禨
		class_list.append(np.argmin(distances))# 将嵌入向量标记为与其距离最近的原型向量所属的类别
	return class_list

def freeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = False

def unfreeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = True

def load_diffusion_model(folder, fname, modelname):
    """加载扩散模型"""
    path = os.path.join(folder, fname)
    model = eval(modelname + '()')
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
    epoch = -1
    accuracy_list = []
    
    if os.path.exists(path):
        try:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            accuracy_list = checkpoint['accuracy_list']
            print(f"加载扩散模型成功: {modelname}, 轮次: {epoch}")
        except Exception as e:
            print(f"加载扩散模型失败: {e}")
    else:
        print(f"扩散模型文件不存在: {path}，将使用未训练模型")
    
    return model, None, optimizer, None, epoch, accuracy_list

class color:
	HEADER = '\033[95m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	RED = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'