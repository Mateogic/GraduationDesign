from .Workload import *
from simulator.container.IPSModels.IPSMBitbrain import *
from simulator.container.RAMModels.RMBitbrain import *
from simulator.container.DiskModels.DMBitbrain import *
from random import gauss, randint, Random
from os import path, makedirs, listdir, remove
import wget
from zipfile import ZipFile
import shutil
import pandas as pd
import warnings
import os
import datetime
warnings.simplefilter("ignore")

# Intel Pentium III gives 2054 MIPS at 600 MHz
# Source: https://archive.vn/20130205075133/http://www.tomshardware.com/charts/cpu-charts-2004/Sandra-CPU-Dhrystone,449.html
ips_multiplier = 2054.0 / (2 * 600)

class BWGD2(Workload):
    def __init__(self, meanNumContainers, sigmaNumContainers, seed):
        super().__init__()
        self.mean = meanNumContainers
        self.sigma = sigmaNumContainers

        # 初始化随机数生成器并设置种子
        self.seed = seed
        self.random_generator = Random(seed)

        dataset_path = 'simulator/workload/datasets/bitbrain/'
        rnd_path = 'simulator/workload/rnd/'
        if not path.exists(dataset_path):
            makedirs(dataset_path)
            shutil.move(rnd_path, dataset_path)
            # print('Downloading Bitbrain Dataset')
            # url = 'http://gwa.ewi.tudelft.nl/fileadmin/pds/trace-archives/grid-workloads-archive/datasets/gwa-t-12/rnd.zip'
            # filename = wget.download(url); zf = ZipFile(filename, 'r'); zf.extractall(dataset_path); zf.close()
            for f in listdir(dataset_path+'rnd/2013-9/'): shutil.move(dataset_path+'rnd/2013-9/'+f, dataset_path+'rnd/')
            shutil.rmtree(dataset_path+'rnd/2013-7'); shutil.rmtree(dataset_path+'rnd/2013-8')
            shutil.rmtree(dataset_path+'rnd/2013-9')
        self.dataset_path = dataset_path
        self.disk_sizes = [1, 2, 3]
        self.meanSLA, self.sigmaSLA = 20, 3
        self.possible_indices = []
        for i in range(1, 500):
            df = pd.read_csv(self.dataset_path+'rnd/'+str(i)+'.csv', sep=';\t')
            if (ips_multiplier*df['CPU usage [MHZ]']).to_list()[10] < 3000 and (ips_multiplier*df['CPU usage [MHZ]']).to_list()[10] > 500:
                self.possible_indices.append(i)

        # 创建日志目录
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # 创建CSV选择记录文件
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file_path = f"{logs_dir}/csv_selection_seed{seed}_{timestamp}.log"

        # 写入日志文件头
        with open(self.log_file_path, 'w') as f:
            f.write(f"# BitbrainWorkload CSV选择记录 - 种子: {seed}\n")
            f.write(f"# 创建时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# 格式: 时间间隔,容器ID,CSV文件索引,SLA值\n")
            f.write("-" * 50 + "\n")

    def generateNewContainers(self, interval):
        workloadlist = []
        # 计算本次生成的容器数量
        num_containers = max(1, int(self.random_generator.gauss(self.mean, self.sigma)))

        # 打开日志文件
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"\n时间间隔 {interval} - 生成 {num_containers} 个容器:\n")

            # 使用随机生成器的方法代替全局函数
            for i in range(num_containers):
                CreationID = self.creation_id
                # 使用独立的随机生成器选择索引
                index_position = self.random_generator.randint(0, len(self.possible_indices)-1)
                index = self.possible_indices[index_position]
                df = pd.read_csv(self.dataset_path+'rnd/'+str(index)+'.csv', sep=';\t')
                # 使用随机生成器生成SLA
                sla = self.random_generator.gauss(self.meanSLA, self.sigmaSLA)

                # 记录选择的CSV文件
                log_file.write(f"  容器ID: {CreationID}, CSV文件: {index}, SLA: {sla:.2f}\n")

                IPSModel = IPSMBitbrain((ips_multiplier*df['CPU usage [MHZ]']).to_list(),
                    (ips_multiplier*df['CPU capacity provisioned [MHZ]']).to_list()[0],
                    int(1.2*sla), interval + sla)
                RAMModel = RMBitbrain((df['Memory usage [KB]']/4000).to_list(),
                    (df['Network received throughput [KB/s]']/1000).to_list(),
                    (df['Network transmitted throughput [KB/s]']/1000).to_list())
                disk_size = self.disk_sizes[index % len(self.disk_sizes)]
                DiskModel = DMBitbrain(disk_size,
                    (df['Disk read throughput [KB/s]']/4000).to_list(),
                    (df['Disk write throughput [KB/s]']/12000).to_list())

                workloadlist.append((CreationID, interval, IPSModel, RAMModel, DiskModel))
                self.creation_id += 1

        self.createdContainers += workloadlist
        self.deployedContainers += [False] * len(workloadlist)
        return self.getUndeployedContainers()