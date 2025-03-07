import os
import torch
import numpy as np
import pandas as pd
from recovery.PreGANSrc.data.simulator.convert import load_dataset, data_filename, normalize_time_data


def save_2d_to_csv(data, filename):
    """将二维数组保存为CSV文件"""
    if torch.is_tensor(data):
        data = data.numpy()
    pd.DataFrame(data).to_csv(filename, index=False, header=False)


def save_3d_to_csv(data, filename):
    """将三维数组保存为CSV文件，每个2D矩阵之间有一个空行"""
    if torch.is_tensor(data):
        data = data.numpy()

    with open(filename, 'w') as f:
        for i in range(data.shape[0]):
            if i > 0:
                f.write('\n')  # 矩阵之间添加空行
            np.savetxt(f, data[i], delimiter=',', fmt='%.6f')


def save_all_data():
    """保存所有数据为CSV文件"""
    # 加载数据集
    train_time_data, train_schedule_data, anomaly_data, class_data = load_dataset()

    # 获取原始time_data
    time_data = np.load(data_filename)
    time_data = normalize_time_data(time_data)

    print(f"数据形状信息:")
    print(f"time_data: {time_data.shape}")
    print(f"train_time_data: {train_time_data.shape}")
    print(f"train_schedule_data: {train_schedule_data.shape}")
    print(f"anomaly_data: {anomaly_data.shape}")
    print(f"class_data: {class_data.shape}")

    # 创建输出目录
    os.makedirs('data_csv', exist_ok=True)

    # 保存数据
    save_2d_to_csv(time_data, 'data_csv/time_data.csv')
    save_3d_to_csv(train_time_data, 'data_csv/train_time_data.csv')
    save_3d_to_csv(train_schedule_data, 'data_csv/train_schedule_data.csv')
    save_2d_to_csv(anomaly_data, 'data_csv/anomaly_data.csv')
    save_2d_to_csv(class_data, 'data_csv/class_data.csv')

    print("所有数据已保存为CSV文件在 'data_csv' 目录中")


if __name__ == '__main__':
    save_all_data()