from .Workload import *
from simulator.container.IPSModels.IPSMBitbrain import *
from simulator.container.RAMModels.RMBitbrain import *
from simulator.container.DiskModels.DMBitbrain import *
from random import gauss, randint
from os import path, makedirs, listdir, remove
import wget
from zipfile import ZipFile
import shutil
import pandas as pd
import warnings
warnings.simplefilter("ignore")

# Intel Pentium III gives 2054 MIPS at 600 MHz
# Source: https://archive.vn/20130205075133/http://www.tomshardware.com/charts/cpu-charts-2004/Sandra-CPU-Dhrystone,449.html
ips_multiplier = 2054.0 / (2 * 600)

class BWGD2(Workload):
	def __init__(self, meanNumContainers, sigmaNumContainers):
		super().__init__()
		self.mean = meanNumContainers
		self.sigma = sigmaNumContainers
		dataset_path = 'simulator/workload/datasets/bitbrain/'
		rnd_path = 'simulator/workload/rnd/'
		if not path.exists(dataset_path):
			makedirs(dataset_path)
			shutil.move(rnd_path, dataset_path)# 复制rnd_path目录内容到dataset_path
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

	def generateNewContainers(self, interval):# 生成新的容器信息
		workloadlist = []# 用于存储生成容器的信息
		for i in range(max(1,int(gauss(self.mean, self.sigma)))):# 根据高斯分布生成随机数确定生成容器的数量，至少为1
			CreationID = self.creation_id# 获取当前创建的容器ID
			index = self.possible_indices[randint(0,len(self.possible_indices)-1)]# 随机选择一个可能的索引
			df = pd.read_csv(self.dataset_path+'rnd/'+str(index)+'.csv', sep=';\t')# 读取索引对应的csv文件，获取容器的资源使用数据
			sla = gauss(self.meanSLA, self.sigmaSLA)# 根据高斯分布生成随机数确定SLA(服务级别协议)
			# 创建容器的IPSModel、RAMModel、DiskModel对象，分别表示容器的CPU、内存、磁盘资源使用情况
			IPSModel = IPSMBitbrain((ips_multiplier*df['CPU usage [MHZ]']).to_list(), (ips_multiplier*df['CPU capacity provisioned [MHZ]']).to_list()[0], int(1.2*sla), interval + sla)
			RAMModel = RMBitbrain((df['Memory usage [KB]']/4000).to_list(), (df['Network received throughput [KB/s]']/1000).to_list(), (df['Network transmitted throughput [KB/s]']/1000).to_list())
			disk_size  = self.disk_sizes[index % len(self.disk_sizes)]
			DiskModel = DMBitbrain(disk_size, (df['Disk read throughput [KB/s]']/4000).to_list(), (df['Disk write throughput [KB/s]']/12000).to_list())
			workloadlist.append((CreationID, interval, IPSModel, RAMModel, DiskModel))# 将容器信息添加到workloadlist中
			self.creation_id += 1
		self.createdContainers += workloadlist# 将workloadlist中的容器信息添加到createdContainers中
		self.deployedContainers += [False] * len(workloadlist)# 为每个新生成的容器添加一个False值，表示未部署
		return self.getUndeployedContainers()# 返回尚未部署的容器信息