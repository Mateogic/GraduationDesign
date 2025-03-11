import os
import sys
import glob
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from .src.models import Diffusion_16
from .src.utils import save_model
from torch.distributions import Beta
sys.path.append('recovery/PreDiffusionSrc/datasets/')
class DiffusionDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.weights = []
        
        # 加载所有数据文件
        data_files = glob.glob(os.path.join(data_dir, 'diffusion_data_*.pkl'))
        print(f"Found {len(data_files)} data files")
        
        # 从所有文件加载数据
        for file_path in tqdm(data_files, desc="Loading dataset"):
            try:
                with open(file_path, 'rb') as f:
                    batch_samples = pickle.load(f)
                    
                for sample in batch_samples:
                    # 计算样本权重(性能提升越大，权重越高)
                    weight = max(0.1, sample['performance_gain'] * 10 + 1.0)
                    
                    # 如果是高质量样本，增加权重
                    if sample['new_score'] < sample['orig_score']:
                        weight *= 2.0
                        
                    self.samples.append(sample)
                    self.weights.append(weight)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        # 归一化权重
        if self.weights:
            total = sum(self.weights)
            self.weights = [w/total for w in self.weights]
            
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'embedding': sample['embedding'],
            'schedule_data': sample['schedule_data'],
            'delta': sample['delta'],
            'weight': torch.tensor(self.weights[idx])
        }

def train_diffusion_model():
    # 参数设置
    batch_size = 32
    num_epochs = 100
    lr = 0.0001
    
    # 准备数据集
    dataset = DiffusionDataset('recovery/PreDiffusionSrc/datasets')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)# 打破时间相关性，提高模型的泛化能力
    
    # 初始化模型
    model = Diffusion_16().double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    
    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            # 准备输入
            embedding = batch['embedding']
            schedule_data = batch['schedule_data']  
            target_delta = batch['delta']
            weights = batch['weight']
            
            # 不需要前向传播，而是直接使用模型的训练逻辑
            loss = 0
            batch_size = embedding.shape[0]
            
            for i in range(batch_size):
                # 使用Beta分布采样，更关注中间阶段噪声
                t = torch.distributions.Beta(torch.tensor([2.0]), torch.tensor([2.0])).sample().double()
                
                # 为目标增量添加噪声
                target = target_delta[i].view(-1)
                noise = torch.randn_like(target)
                
                # 计算t时刻的带噪声数据
                index = int(t.item() * (model.n_steps - 1))
                alpha_cumprod_t = model.alphas_cumprod[index]
                x_t = torch.sqrt(alpha_cumprod_t) * target + torch.sqrt(1 - alpha_cumprod_t) * noise
                
                # 条件信息
                cond = torch.cat((embedding[i].view(-1), schedule_data[i].view(-1)))
                
                # 预测噪声
                noise_pred = model._denoise(x_t, t, cond)
                
                # 1. 标准噪声预测损失
                noise_loss = mse_loss(noise_pred, noise)
                
                # 2. 直接预测无噪声目标损失(x0预测)
                # 从噪声预测还原原始信号
                pred_clean = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
                direct_loss = mse_loss(pred_clean, target)
                
                # 根据训练进度动态调整权重
                epoch_progress = epoch / num_epochs
                noise_weight = 0.7 - 0.2 * epoch_progress  # 从0.7降到0.5
                direct_weight = 0.3 + 0.2 * epoch_progress  # 从0.3升到0.5
                
                # 组合损失
                sample_loss = (noise_weight * noise_loss + direct_weight * direct_loss) * weights[i]
                loss += sample_loss
                
            # 平均损失
            loss /= batch_size
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 每个epoch结束打印信息
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
        
        # 保存模型
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            save_model('recovery/PreDiffusionSrc/checkpoints/prediffusion', f'simulator_Diffusion_16.ckpt', model, optimizer, epoch, [(epoch, avg_loss)])

if __name__ == "__main__":
    train_diffusion_model()