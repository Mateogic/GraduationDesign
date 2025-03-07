import os, sys, stat
import sys
import optparse
import logging as logger
import configparser
import pickle
import shutil
import sqlite3
import platform
from time import time
from subprocess import call
from os import system, rename

# Framework imports
from framework.Framework import *
from framework.database.Database import *
from framework.datacenter.Datacenter_Setup import *
from framework.datacenter.Datacenter import *
from framework.workload.DeFogWorkload import *
from recovery.GOBI import GOBIRecovery

# Simulator imports
from simulator.Simulator import *
from simulator.environment.AzureFog import *
from simulator.environment.RPiEdge import *
from simulator.environment.BitbrainFog import *
from simulator.workload.BitbrainWorkload_GaussianDistribution import *
from simulator.workload.BitbrainWorkload2 import *

# Scheduler imports
from scheduler.IQR_MMT_Random import IQRMMTRScheduler
from scheduler.MAD_MMT_Random import MADMMTRScheduler
from scheduler.MAD_MC_Random import MADMCRScheduler
from scheduler.LR_MMT_Random import LRMMTRScheduler
from scheduler.Random_Random_FirstFit import RFScheduler
from scheduler.Random_Random_LeastFull import RLScheduler
from scheduler.RLR_MMT_Random import RLRMMTRScheduler
from scheduler.Threshold_MC_Random import TMCRScheduler
from scheduler.Random_Random_Random import RandomScheduler
from scheduler.HGP_LBFGS import HGPScheduler
from scheduler.GA import GAScheduler
from scheduler.GOBI import GOBIScheduler
from scheduler.GOBI2 import GOBI2Scheduler
from scheduler.DRL import DRLScheduler
from scheduler.DQL import DQLScheduler
from scheduler.POND import PONDScheduler
from scheduler.SOGOBI import SOGOBIScheduler
from scheduler.SOGOBI2 import SOGOBI2Scheduler
from scheduler.HGOBI import HGOBIScheduler
from scheduler.HGOBI2 import HGOBI2Scheduler
from scheduler.HSOGOBI import HSOGOBIScheduler
from scheduler.HSOGOBI2 import HSOGOBI2Scheduler

# Recovery imports
from recovery.Recovery import Recovery
from recovery.PreGAN import PreGANRecovery
from recovery.PreGANPlus import PreGANPlusRecovery
from recovery.PCFT import PCFTRecovery
from recovery.DFTM import DFTMRecovery
from recovery.ECLB import ECLBRecovery
from recovery.CMODLB import CMODLBRecovery

# Auxiliary imports
from stats.Stats import *
from utils.Utils import *
from pdb import set_trace as bp
import torch
# usage 变量定义了脚本的使用说明，提示用户如何运行脚本以及需要提供的参数。
usage = "usage: python main.py -e <environment> -m <mode> # empty environment run simulator"
# 创建一个 OptionParser 对象 parser，并将 usage 变量传递给它，用于显示帮助信息。
parser = optparse.OptionParser(usage=usage)
# 使用 add_option 方法为 parser 添加两个命令行选项：
# -e指定环境，存储在 opts.env 中，默认值为空字符串。帮助信息说明了可选的环境值
# -m指定模式，存储在 opts.mode 中，默认值为0。帮助信息说明了可选的模式值
parser.add_option("-e", "--environment", action="store", dest="env", default="", 
					help="Environment is AWS, Openstack, Azure, VLAN, Vagrant")
parser.add_option("-m", "--mode", action="store", dest="mode", default="0", 
					help="Mode is 0 (Create and destroy), 1 (Create), 2 (No op), 3 (Destroy)")
# 调用 parse_args 方法解析命令行参数，返回一个元组，元组中包含两个元素，第一个元素是opts，是一个包含所有选项值的对象，第二个元素是args，是一个包含所有非选项参数的列表。
opts, args = parser.parse_args()

# 全局变量
NUM_SIM_STEPS = 50# 模拟轮数
HOSTS = 16 if opts.env == '' else 16# 主机数
CONTAINERS = HOSTS# 容器数=主机数
TOTAL_POWER = 1000# 总功率
ROUTER_BW = 10000# 路由器带宽
INTERVAL_TIME = 300 # 间隔时间(ms)
NEW_CONTAINERS = 1# 新容器数 泊松分布lambda
SEED = 3407# 随机数种子
# 好种子：3407
# 坏种子：

DB_NAME = ''# 数据库名称
DB_HOST = ''# 数据库主机地址
DB_PORT = 0# 数据库端口
HOSTS_IP = []# 主机IP地址
logFile = 'COSCO.log'# 日志文件

if len(sys.argv) > 1:# 检查是否提供命令行参数，若有
	with open(logFile, 'w'): os.utime(logFile, None)# 打开/创建日志文件

def initalizeEnvironment(environment, logger):
	if environment != '':# 如果 environment 参数不为空，则初始化数据库连接。
		db = Database(DB_NAME, DB_HOST, DB_PORT)

	# 如果 environment 参数不为空，则初始化虚拟雾数据中心。
	# Initialize simple fog datacenter
	''' Can be SimpleFog, BitbrainFog, AzureFog // Datacenter '''
	if environment != '':
		datacenter = Datacenter(HOSTS_IP, environment, 'Virtual')
	else:
		datacenter = RPiEdge(HOSTS)# 初始化简单的边缘数据中心（RPiEdge）

	# Initialize workload
	''' Can be SWSD, BWGD, BWGD2 // DFW '''
	if environment != '':
		workload = DFW(NEW_CONTAINERS, 1.5, db)
	else: # 使用rnd数据集初始化 BWGD2 工作负载
		workload = BWGD2(NEW_CONTAINERS, 1.5, seed = SEED)# NEW_CONTAINERS = 1
	
	# Initialize scheduler
	# 初始化 GOBI 调度器，参数为 energy_latency_ 加上主机数量。
	''' Can be LRMMTR, RF, RL, RM, Random, RLRMMTR, TMCR, TMMR, TMMTR, GA, GOBI (arg = 'energy_latency_'+str(HOSTS)) '''
	scheduler = GOBIScheduler('energy_latency_'+str(HOSTS))

	# Initialize recovery
	# 初始化 PreGANPlus 恢复机制，传入主机数量、环境和训练标志
	''' Can be PreGANPlusRecovery, PreGANRecovery, CMODLBRecovery, PCFTRecovery, ECLBRecovery, DFTMRecovery, GOBIRecovery '''
	recovery = PreGANPlusRecovery(HOSTS, environment, training = True)

	# Initialize Stats
	# 初始化统计信息对象，传入工作负载、数据中心和调度器。
	stats = Stats(workload, datacenter, scheduler)

	# Initialize Environment
	# 初始化环境对象，传入调度器、恢复机制、统计信息、容器数量、间隔时间和主机列表。
	hostlist = datacenter.generateHosts()
	if environment != '':
		env = Framework(scheduler, recovery, stats, CONTAINERS, INTERVAL_TIME, hostlist, db, environment, logger)
	else:# 初始化模拟器环境
		env = Simulator(TOTAL_POWER, ROUTER_BW, scheduler, recovery, stats, CONTAINERS, INTERVAL_TIME, hostlist)

	# Execute first step
	torch.compile()# 编译 PyTorch 模型
	newcontainerinfos = workload.generateNewContainers(env.interval) # 生成新容器信息
	deployed = env.addContainersInit(newcontainerinfos) # 部署新容器并且获取容器 ID
	start = time()
	decision = scheduler.placement(deployed) # Decide placement using container ids(x,y)表示将容器x放在主机y上
	schedulingTime = time() - start# 记录调度时间
	migrations = env.allocateInit(decision) # 调度容器动作，将容器x部署到主机y上
	workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed)) # Update workload allocated using creation IDs
	print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))# 打印部署容器的创建 ID [0]
	print("Containers in host:", env.getContainersInHosts())# 打印主机中的容器 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
	print("Schedule:", env.getActiveContainerList())# [14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]表示容器0被分配到主机14，其他容器未被分配
	printDecisionAndMigrations(decision, migrations)

	# 保存统计信息，包括部署、迁移、销毁、选择、决策和调度时间。为啥有两个deployed？
	stats.saveStats(deployed, migrations, [], deployed, decision, schedulingTime)
	# 返回数据中心、工作负载、调度决策、恢复机制、环境和统计信息。
	return datacenter, workload, scheduler, recovery, env, stats

def stepSimulation(workload, scheduler, recovery, env, stats):
	newcontainerinfos = workload.generateNewContainers(env.interval)# 生成新容器信息
	if opts.env != '': print(newcontainerinfos)
	deployed, destroyed = env.addContainers(newcontainerinfos)# 销毁的容器和要部署的容器id
	start = time()
	selected = scheduler.selection() # GOBI始终返回[]
	decision = scheduler.filter_placement(scheduler.placement(selected+deployed)) # 原始决策S-利用GOBI调度器生成决策并根据当前主机和目标主机是否相同过滤不必要的迁移决策
	schedulingTime = time() - start# 记录调度时间
	recovered_decision = recovery.run_model(stats.time_series, decision)# PreGAN+模型在线推理得到的决策N
	migrations = env.simulationStep(recovered_decision) # 根据容器的需求和主机的资源判断是否可以迁移，仅保留合理的迁移决策
	workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed)) # 使用创建ID 更新工作负载的部署信息
	print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))# 打印部署容器的创建 ID
	print("Deployed:", len(env.getCreationIDs(migrations, deployed)), "of", len(newcontainerinfos), [i[0] for i in newcontainerinfos])# 打印已部署容器的数量和新的容器信息
	print("Destroyed:", len(destroyed), "of", env.getNumActiveContainers())# 打印销毁的容器数量和活动容器数量
	print("Containers in host:", env.getContainersInHosts())# 打印主机上的容器数量
	print("Num active containers:", env.getNumActiveContainers())# 打印活动容器数量
	print("Host allocation:", [(c.getHostID() if c else -1)for c in env.containerlist])# 打印容器分配到主机的情况
	printDecisionAndMigrations(decision, migrations)# 打印决策和迁移信息 红色表示存在于decision而不存在于migrations

	stats.saveStats(deployed, migrations, destroyed, selected, decision, schedulingTime)# 保存统计信息

# 保存统计信息
def saveStats(stats, datacenter, workload, env, end=True):
	# 构建日志目录名
	dirname = "logs/" + datacenter.__class__.__name__
	dirname += "_" + workload.__class__.__name__
	dirname += "_" + str(NUM_SIM_STEPS) 
	dirname += "_" + str(HOSTS)
	dirname += "_" + str(CONTAINERS)
	dirname += "_" + str(TOTAL_POWER)
	dirname += "_" + str(ROUTER_BW)
	dirname += "_" + str(INTERVAL_TIME)
	dirname += "_" + str(NEW_CONTAINERS)
	if not os.path.exists("logs"): os.mkdir("logs")
	# 如果目录存在，则删除目录
	if os.path.exists(dirname): shutil.rmtree(dirname, ignore_errors=True)
	os.mkdir(dirname)
	# 生成数据集
	stats.generateDatasets(dirname)
	if 'Datacenter' in datacenter.__class__.__name__:
		saved_env, saved_workload, saved_datacenter, saved_scheduler, saved_sim_scheduler = stats.env, stats.workload, stats.datacenter, stats.scheduler, stats.simulated_scheduler
		stats.env, stats.workload, stats.datacenter, stats.scheduler, stats.simulated_scheduler = None, None, None, None, None
		with open(dirname + '/' + dirname.split('/')[1] +'.pk', 'wb') as handle:
			pickle.dump(stats, handle)
		stats.env, stats.workload, stats.datacenter, stats.scheduler, stats.simulated_scheduler = saved_env, saved_workload, saved_datacenter, saved_scheduler, saved_sim_scheduler
	if not end: return
	stats.generateGraphs(dirname)
	stats.generateCompleteDatasets(dirname)
	stats.env, stats.workload, stats.datacenter, stats.scheduler = None, None, None, None
	if 'Datacenter' in datacenter.__class__.__name__:
		stats.simulated_scheduler = None
		logger.getLogger().handlers.clear(); env.logger.getLogger().handlers.clear()
		if os.path.exists(dirname+'/'+logFile): os.remove(dirname+'/'+logFile)
		rename(logFile, dirname+'/'+logFile)
	with open(dirname + '/' + dirname.split('/')[1] +'.pk', 'wb') as handle:
	    pickle.dump(stats, handle)

if __name__ == '__main__':
	env, mode = opts.env, int(opts.mode)# 获取环境和模式

	if env != '':
		# Convert all agent files to unix format
		unixify(['framework/agent/', 'framework/agent/scripts/'])

		# Start InfluxDB service
		print(color.HEADER+'InfluxDB service runs as a separate front-end window. Please minimize this window.'+color.ENDC)
		if 'Windows' in platform.system():
			os.startfile('C:/Program Files/InfluxDB/influxdb-1.8.3-1/influxd.exe')

		configFile = 'framework/config/' + opts.env + '_config.json'
	    
		logger.basicConfig(filename=logFile, level=logger.DEBUG,
	                        format='%(asctime)s - %(levelname)s - %(message)s')
		logger.debug("Creating enviornment in :{}".format(env))
		cfg = {}
		with open(configFile, "r") as f:
			cfg = json.load(f)
		DB_HOST = cfg['database']['ip']
		DB_PORT = cfg['database']['port']
		DB_NAME = 'COSCO'

		if env == 'Vagrant':
			print("Setting up VirtualBox environment using Vagrant")
			HOSTS_IP = setupVagrantEnvironment(configFile, mode)
			print(HOSTS_IP)
		elif env == 'VLAN':
			print("Setting up VLAN environment using Ansible")
			HOSTS_IP = setupVLANEnvironment(configFile, mode)
			print(HOSTS_IP)
		# exit()
	# 无参执行起点
	datacenter, workload, scheduler, recovery, env, stats = initalizeEnvironment(env, logger)

	for step in range(NUM_SIM_STEPS):# 循环模拟
		print(color.BOLD+("Simulation" if opts.env == '' else "Execution")+" Interval:", step, color.ENDC)# 打印模拟次数
		stepSimulation(workload, scheduler, recovery, env, stats)
		if env != '' and step % 10 == 0: saveStats(stats, datacenter, workload, env, end = False)

	if opts.env != '':
		# Destroy environment if required
		eval('destroy'+opts.env+'Environment(configFile, mode)')

		# Quit InfluxDB
		if 'Windows' in platform.system():
			os.system('taskkill /f /im influxd.exe')

	saveStats(stats, datacenter, workload, env)

