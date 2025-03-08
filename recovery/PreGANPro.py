import sys

sys.path.append('recovery/PreGANSrc/')

import numpy as np
from copy import deepcopy
from .Recovery import *
from .PreGANSrc.src.constants import *
from .PreGANSrc.src.utils import *
from .PreGANSrc.src.train import *
from collections import deque

class PreGANProRecovery(Recovery):# 继承关系
    def __init__(self, hosts, env, training = False):# 构造函数类实例化时自动调用
        super().__init__()# 其中初始化了self.model
        self.model_name = f'TransformerPro_{hosts}'
        self.gen_name = f'Gen_{hosts}'
        self.disc_name = f'Disc_{hosts}'
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.training = training
        self.save_gan = True
        self.model = TransformerPro_16()
        self.maxLen = self.model.n_window# 历史嵌入和时间窗口大小相同
        self.flag = False  # 用于标记是否清空过accuracy_list和epoch(FPE训练结束之后)
        self.historical_embeddings = deque(maxlen=self.maxLen)  # 保留历史嵌入
        self.load_models()

    def load_models(self):# 在构造函数中调用此函数加载模型
        # Load encoder model
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_pro_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)
        # Train the model if not trained (offline training same as PreGAN)
        if self.epoch == -1: self.train_model()
        # Reduce lr of encoder
        # self.model.lr /= 5
        # Load generator and discriminator
        self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list = \
            load_gan(model_pro_folder, f'{self.env_name}_{self.gen_name}.ckpt', f'{self.env_name}_{self.disc_name}.ckpt', self.gen_name, self.disc_name)
        self.gan_plotter = GAN_Plotter(self.env_name, self.gen_name, self.disc_name, self.training)
        # GAN is always tuned
        self.ganloss = nn.BCELoss()
        # 加载time_series.npy
        self.train_time_data = load_npyfile(os.path.join(data_folder, self.env_name), data_filename)

    def train_model(self):# checkpointspro目录下没有模型文件时self.epoch == -1，调用此函数离线训练PreGAN+
        self.model_plotter = Model_Plotter(self.env_name, self.model_name)
        folder = os.path.join(data_folder, self.env_name)
        # 加载四种数据集
        train_time_data, train_schedule_data, anomaly_data, class_data = load_dataset(folder, self.model)
        for self.epoch in tqdm(range(self.epoch+1, self.epoch+num_epochs+1), position=0):
            loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.optimizer)
            anomaly_score, class_score = accuracy(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.model_plotter)
            tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}\n')
            self.accuracy_list.append((loss, factor, anomaly_score, class_score))
            self.model_plotter.plot(self.accuracy_list, self.epoch)# 绘图
            save_model(model_pro_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model, self.optimizer, self.epoch, self.accuracy_list)

    def tune_model(self):# 在线微调PreGAN+
        # tune for a single epoch
        folder = os.path.join(data_folder, self.env_name)
        train_time_data, train_schedule_data, anomaly_data, class_data = load_on_the_fly_dataset(self.model, folder, self.env.stats)
        loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.optimizer)
        anomaly_score, class_score = accuracy(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, None)
        tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
        # self.accuracy_list.append((loss, factor, anomaly_score, class_score))

    def train_gan(self, embedding, schedule_data):# 在线训练GAN
        # Train discriminator
        self.disc.zero_grad()
        new_schedule_data = self.gen(embedding, schedule_data)
        probs = self.disc(schedule_data, new_schedule_data.detach())
        new_score, orig_score = run_simulation(self.env.stats, new_schedule_data), run_simulation(self.env.stats, schedule_data)
        true_probs = torch.tensor([0, 1], dtype=torch.double) if new_score <= orig_score else torch.tensor([1, 0], dtype=torch.double)
        disc_loss = self.ganloss(probs, true_probs.detach().clone())
        disc_loss.backward(); self.dopt.step()
        # Train generator
        self.gen.zero_grad()
        probs = self.disc(schedule_data, new_schedule_data)
        true_probs = torch.tensor([0, 1], dtype=torch.double) # to enforce new schedule is better than original schedule
        gen_loss = self.ganloss(probs, true_probs)
        gen_loss.backward(); self.gopt.step()
        # Append to accuracy list and save model
        if self.save_gan:
            self.epoch += 1; self.accuracy_list.append((gen_loss.item(), disc_loss.item()))
            tqdm.write(f'Epoch {self.epoch},\tGLoss = {gen_loss.item()},\tDLoss = {disc_loss.item()}')
            self.gan_plotter.plot(self.accuracy_list, self.epoch, new_score, orig_score)# 绘图
            save_gan(model_pro_folder, f'{self.env_name}_{self.gen_name}.ckpt', f'{self.env_name}_{self.disc_name}.ckpt', \
                    self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list)

    def recover_decision(self, embedding, schedule_data, original_decision):# 在线推理
        new_schedule_data = self.gen(embedding, schedule_data)# N = Gen(E^F,S)
        probs = self.disc(schedule_data, new_schedule_data)# D = Disc(S,N)
        self.gan_plotter.new_better(probs[1] >= probs[0])
        if probs[0] > probs[1]: # original better
            return original_decision
        # Form new decision
        host_alloc = []; container_alloc = [-1] * len(self.env.hostlist)
        for i in range(len(self.env.hostlist)): host_alloc.append([])
        for c in self.env.containerlist:# 复制主机和容器的分配情况
            if c and c.getHostID() != -1:# c.getHostID() != -1表示该容器已部署到某台主机
                host_alloc[c.getHostID()].append(c.id)
                container_alloc[c.id] = c.getHostID()
        decision_dict = dict(original_decision); hosts_from = [0] * self.hosts
        for cid in np.concatenate(host_alloc):# 数组拼接操作
            cid = int(cid)
            one_hot = new_schedule_data[cid].tolist()# 取出new_schedule_data的第cid行形成列表
            # one_hot = schedule_data[cid].tolist()
            new_host = one_hot.index(max(one_hot))# 列表中最大值所在索引
            if container_alloc[cid] != new_host: # cid对应容器原本部署的主机与新的目标主机不一致，新增迁移条目(cid,new_host)到decision_dict
                decision_dict[cid] = new_host
                hosts_from[container_alloc[cid]] = 1
        self.gan_plotter.plot_test(hosts_from, self.epoch)
        return list(decision_dict.items())

    def run_encoder(self, schedule_data):
        # 获取最新数据
        time_data = self.env.stats.time_series
        time_data = normalize_test_time_data(time_data, self.train_time_data)
        if time_data.shape[0] >= self.model.n_window:
            time_data = time_data[-self.model.n_window:]
        time_data = convert_to_windows(time_data, self.model)[-1]
        
        # 使用历史嵌入运行模型
        anomaly, prototype, current_gat = self.model(time_data, schedule_data, 
                                                historical_embeddings=self.historical_embeddings,
                                                return_embedding=True)
        
        # 保存当前GAT结果到历史嵌入
        self.historical_embeddings.append(current_gat.detach().clone())
        
        return anomaly, prototype

    def run_model(self, time_series, original_decision):
        # 清空训练FPE时的accuracy_list
        if(not self.flag):
            self.accuracy_list = []
            self.epoch = 0
            self.flag = True
            self.historical_embeddings = deque(maxlen=self.maxLen)  # 重置历史嵌入队列
        # Run encoder
        schedule_data = torch.tensor(self.env.scheduler.result_cache).double()# S 16*16
        anomaly, prototype = self.run_encoder(schedule_data)# D(16*1*2, 预测每个主机是否故障), P
        # If no anomaly predicted, return original decision 
        for a in anomaly:
            prediction = torch.argmax(a).item() # 找到最大值所在索引
            if prediction == 1: # 有故障，进入GAN尝试生成新决策
                self.gan_plotter.update_anomaly_detected(1)
                break
        else:# 无故障，返回原决策
            self.gan_plotter.update_anomaly_detected(0)
            return original_decision
        # Form prototype vectors for diagnosed hosts
        embedding = [torch.zeros_like(p) if torch.argmax(anomaly[i]).item() == 0 else p for i, p in enumerate(prototype)]
        self.gan_plotter.update_class_detected(get_classes(embedding, self.model))
        embedding = torch.stack(embedding)# 堆叠后的
        # Pass through GAN self.epoch+=1(预测有故障时)
        self.train_gan(embedding, schedule_data)# Epoch 76,       GLoss = 0.23239809802868833,    DLoss = 0.23744803135508769
        # Tune Model
        self.tune_model()# Epoch 76,        Loss = 17.7339489329655,        ALoss = 18.193004121730326,     TLoss = -0.45905518876482454
        return self.recover_decision(embedding, schedule_data, original_decision)# 在线推理