import os
import torch
import numpy as np

data_filename = 'time_series.npy'# 离线训练FPE模型时 和 在线微调FPE时加载
schedule_filename = 'schedule_series.npy'# 离线训练FPE模型时加载

def load_dataset():
	time_data = np.load(data_filename)
	time_data = normalize_time_data(time_data) # Normalize data
	train_schedule_data = torch.tensor(np.load(schedule_filename)).double()
	train_time_data = convert_to_windows(time_data)
	anomaly_data, class_data = form_test_dataset(time_data)
	return train_time_data, train_schedule_data, anomaly_data, class_data

# Misc
def normalize_time_data(time_data):
	return time_data / (np.max(time_data, axis = 0) + 1e-8)

def convert_to_windows(data):
	data = torch.tensor(data).double()
	windows = []; w_size = 10
	for i, g in enumerate(data):
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w)
	return torch.stack(windows)

def form_test_dataset(data):
	anomaly_per_dim = data > np.percentile(data, 98, axis=0)
	anomaly_which_dim, anomaly_any_dim = [], []
	for i in range(0, data.shape[1], 3):
		anomaly_which_dim.append(np.argmax(data[:, i:i+3] + 0, axis=1))
		anomaly_any_dim.append(np.logical_or.reduce(anomaly_per_dim[:, i:i+3], axis=1))
	anomaly_any_dim = np.stack(anomaly_any_dim, axis=1)
	anomaly_which_dim = np.stack(anomaly_which_dim, axis=1)
	return anomaly_any_dim + 0, anomaly_which_dim


if __name__ == '__main__':
	train_time_data, train_schedule_data, anomaly_data, class_data = load_dataset()