import sys

sys.path.append('recovery/PreDiffusionSrc/')
import pickle
import torch
import numpy as np
from copy import deepcopy
from .Recovery import *
from .PreDiffusionSrc.src.constants import *
from .PreDiffusionSrc.src.utils import *
from .PreDiffusionSrc.src.train import *
from collections import deque
from datetime import datetime


class PreDiffusionRecovery(Recovery):# 继承关系
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
        self.maxLen = self.model.n_window * self.model.multi - 1
        self.flag = False  # 用于标记是否清空过accuracy_list和epoch(FPE训练结束之后)
        self.historical_embeddings = deque(maxlen=self.maxLen)  # 保留历史嵌入
        self.recent_performance_gains = deque(maxlen=10)  # 跟踪最近10个样本的性能增益
        self.load_models()

    def load_models(self):# 在构造函数中调用此函数加载模型
        # 加载编码器模型
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)
        # 如未训练，先训练模型
        if self.epoch == -1: self.train_model()
        # 加载Diffusion模型
        print(f"正在加载Diffusion模型: {self.env_name}_Diffusion_{self.hosts}.ckpt")
        self.gen, _, self.gopt, _, self.diff_epoch, self.diff_accuracy_list = \
            load_diffusion_model(model_folder, f'{self.env_name}_Diffusion_{self.hosts}.ckpt', f'Diffusion_{self.hosts}')
        
        # 确保模型处于评估模式
        self.gen.eval()
        
        # 初始化绘图工具
        self.gan_plotter = GAN_Plotter(self.env_name, self.model_name, 'Diffusion', self.training)
        # # GAN is always tuned
        # self.ganloss = nn.BCELoss()
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
            save_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model, self.optimizer, self.epoch, self.accuracy_list)

    def tune_model(self):# 在线微调PreGAN+
        # tune for a single epoch
        folder = os.path.join(data_folder, self.env_name)
        train_time_data, train_schedule_data, anomaly_data, class_data = load_on_the_fly_dataset(self.model, folder, self.env.stats)
        loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.optimizer)
        anomaly_score, class_score = accuracy(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, None)
        tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
        # self.accuracy_list.append((loss, factor, anomaly_score, class_score))

    def generate_schedule(self, embedding, schedule_data):
        """使用Diffusion模型生成新的调度方案"""
        # 使用生成器生成新调度
        with torch.no_grad():  # 推理阶段关闭梯度计算
            new_schedule_data = self.gen(embedding, schedule_data)
        
        # 计算性能得分
        new_score = run_simulation(self.env.stats, new_schedule_data)
        orig_score = run_simulation(self.env.stats, schedule_data)
        
        # 收集数据用于后续训练优化
        self.collect_diffusion_data(embedding, schedule_data, new_schedule_data, new_score, orig_score)

        # 记录结果
        is_better = new_score <= orig_score
        self.gan_plotter.new_better(is_better)
        
        # 记录到日志
        if self.save_gan:
            self.epoch += 1
            self.accuracy_list.append((0.0, 0.0))  # 没有损失值
            tqdm.write(f'Epoch {self.epoch},\tNew Score = {new_score:.4f},\tOrig Score = {orig_score:.4f} {"✓" if is_better else "✗"}')
            self.gan_plotter.plot(self.accuracy_list, self.epoch, new_score, orig_score)
        
        return new_schedule_data, new_score, orig_score, is_better

    def recover_decision(self, embedding, schedule_data, original_decision):
        """使用Diffusion模型生成决策并确定是否采用新决策"""
        # 生成新调度
        new_schedule_data, new_score, orig_score, is_better = self.generate_schedule(embedding, schedule_data)
        
        # 如果新调度性能更差，返回原决策
        if not is_better:
            return original_decision
        
        # 如果新调度更好，形成新决策
        host_alloc = []; container_alloc = [-1] * len(self.env.hostlist)
        for i in range(len(self.env.hostlist)): host_alloc.append([])
        for c in self.env.containerlist:
            if c and c.getHostID() != -1:
                host_alloc[c.getHostID()].append(c.id)
                container_alloc[c.id] = c.getHostID()
        
        decision_dict = dict(original_decision)
        hosts_from = [0] * self.hosts
        for cid in np.concatenate(host_alloc):
            cid = int(cid)
            one_hot = new_schedule_data[cid].tolist()
            new_host = one_hot.index(max(one_hot))
            if container_alloc[cid] != new_host:
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
        if (not self.flag):
            self.accuracy_list = []
            self.epoch = 0
            self.flag = True
            self.historical_embeddings = deque(maxlen=self.maxLen)  # 重置历史嵌入队列

        # 获取当前调度状态
        schedule_data = torch.tensor(self.env.scheduler.result_cache).double()

        # 运行编码器预测故障和主机类型
        anomaly, prototype = self.run_encoder(schedule_data)

        # 检查是否预测到故障
        for a in anomaly:
            prediction = torch.argmax(a).item()
            if prediction == 1:  # 有故障，使用Diffusion模型生成新决策
                self.gan_plotter.update_anomaly_detected(1)
                break
        else:  # 无故障，返回原决策
            self.gan_plotter.update_anomaly_detected(0)
            return original_decision

        # 构建嵌入向量
        embedding = [torch.zeros_like(p) if torch.argmax(anomaly[i]).item() == 0 else p
                     for i, p in enumerate(prototype)]
        self.gan_plotter.update_class_detected(get_classes(embedding, self.model))
        embedding = torch.stack(embedding)
        if self.epoch > 20:
            self.tune_model()# epoch大于20之后在线微调FPE
        # 直接返回恢复决策
        return self.recover_decision(embedding, schedule_data, original_decision)

    # 在PreDiffusionRecovery类中添加以下方法
    def collect_diffusion_data(self, embedding, schedule_data, new_schedule_data, new_score, orig_score):
        """收集PNDM训练数据"""
        # 计算增量决策
        delta = new_schedule_data.detach() - schedule_data.detach()

        # 计算性能提升指标
        performance_gain = (orig_score - new_score) / orig_score if orig_score > 0 else 0
        self.recent_performance_gains.append(performance_gain)  # 添加到历史记录
        # 准备数据样本
        sample = {
            'embedding': embedding.cpu().detach().clone(),
            'schedule_data': schedule_data.cpu().detach().clone(),
            'delta': delta.cpu().detach().clone(),
            'orig_score': float(orig_score),
            'new_score': float(new_score),
            'performance_gain': float(performance_gain)
        }

        # 确保目录存在
        dataset_dir = 'recovery/PreDiffusionSrc/datasets'
        os.makedirs(dataset_dir, exist_ok=True)

        # 样本筛选 - 主要保留高质量样本，但也保留部分次优样本以增强鲁棒性
        should_save = (new_score <= orig_score) or \
                      (new_score > orig_score and (new_score - orig_score) / orig_score < 0.1 and \
                       np.random.random() < 0.2)  # 20%的概率保存次优样本，以增强鲁棒性

        if should_save:
            # 每个文件存储100个样本，避免单文件过大
            sample_count_file = os.path.join(dataset_dir, 'sample_count.txt')
            if os.path.exists(sample_count_file):
                with open(sample_count_file, 'r') as f:
                    count = int(f.read().strip())
            else:
                count = 0

            batch_id = count // 100
            file_path = os.path.join(dataset_dir, f'diffusion_data_{batch_id}.pkl')

            # 追加或创建新文件
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        data_list = pickle.load(f)
                else:
                    data_list = []

                data_list.append(sample)

                with open(file_path, 'wb') as f:
                    pickle.dump(data_list, f)

                # 更新计数
                with open(sample_count_file, 'w') as f:
                    f.write(str(count + 1))

                # 记录数据收集日志
                if count % 10 == 0:  # 每10个样本记录一次
                    avg_gain = sum(self.recent_performance_gains) / len(
                        self.recent_performance_gains) if self.recent_performance_gains else performance_gain
                    tqdm.write(
                        f"{color.RED}Collected diffusion sample #{count + 1}, avg_perf_gain: {avg_gain:.4f}{color.ENDC}")

            except Exception as e:
                print(f"Error saving diffusion data: {e}")