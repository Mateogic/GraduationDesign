import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import statistics
import os, glob
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from .constants import *

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs(plot_folder, exist_ok=True)

def smoother(y, box_pts=1):# box_pts参数调节平滑程度，越大越平滑
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')# mode=same会出现边界效应
	# y = np.array(y)  # Convert to numpy array if it's not already
	# If box_pts is 1 or less, or if the input is too short, return the original data
	# if box_pts <= 1 or len(y) < box_pts:
	# 	return y
	#
	# # Calculate padding
	# pad_left = (box_pts - 1) // 2
	# pad_right = box_pts - 1 - pad_left
	#
	# # Pad the array by repeating edge values to avoid edge effects
	# y_padded = np.pad(y, (pad_left, pad_right), mode='edge')
	#
	# # Create the kernel and apply convolution
	# box = np.ones(box_pts) / box_pts
	# y_smooth = np.convolve(y_padded, box, mode='valid')
    return y_smooth


class Model_Plotter():
	def __init__(self, env, modelname):
		self.env = env
		self.model_name = modelname
		self.n_hosts = int(modelname.split('_')[-1])
		self.folder = os.path.join(plot_folder, env, 'model')
		self.prefix = self.folder + '/' + self.model_name
		self.epoch = 0
		os.makedirs(self.folder, exist_ok=True)
		for f in glob.glob(self.folder + '/*'): os.remove(f)# 删除指定目录下的所有文件
		self.tsne = TSNE(n_components=2, perplexity=50, n_iter=1000)# 将数据非线性降维到2维以便于可视化，perplexity表示数据集中每个点的有效邻居数，通常在5到50之间，n_iter是迭代次数
		self.colors = ['r', 'g', 'b']
		# plt.rcParams["font.family"] = "Maven Pro"
		self.init_params()

	def init_params(self):
		self.source_anomaly_scores = []
		self.target_anomaly_scores = []
		self.correct_series = []
		self.protoypes = []
		self.correct_series_class = []

	def update_anomaly(self, source_anomaly, target_anomaly, correct):
		self.source_anomaly_scores.append(source_anomaly)
		self.target_anomaly_scores.append(target_anomaly.tolist())
		self.correct_series.append(correct)

	def update_class(self, protoypes, correct):
		self.protoypes.extend(protoypes)
		self.correct_series_class.append(correct)

	def plot(self, accuracy_list, epoch):
		self.epoch = epoch; self.prefix2 = self.prefix + '_' + str(self.epoch) + '_'
		self.loss_list = [i[0] for i in accuracy_list]
		self.factor_list = [i[1] for i in accuracy_list]
		self.anomaly_score_list = [i[2] for i in accuracy_list]
		self.class_score_list = [i[3] for i in accuracy_list]
		self.plot1('Loss', self.loss_list)
		self.plot1('Factor', self.factor_list)
		self.plot2('Anomaly Prediction Score', 'Class Prediction Score', self.anomaly_score_list, self.class_score_list)
		self.plot1('Correct Anomaly', self.correct_series, xlabel='Timestamp')
		self.plot1('Correct Class', self.correct_series_class, xlabel='Timestamp')
		self.source_anomaly_scores = np.array(self.source_anomaly_scores)
		self.target_anomaly_scores = np.array(self.target_anomaly_scores)
		self.plot_heatmap('Anomaly Scores', 'Prediction', 'Ground Truth', self.source_anomaly_scores, self.target_anomaly_scores)
		X = [i[0].tolist() for i in self.protoypes]; Y = np.array([i[1] for i in self.protoypes])
		x2d = self.tsne.fit_transform(np.array(X))
		self.plot_tsne('Prototypes', x2d, Y)
		self.init_params()

	def plot1(self, name1, data1, smooth = True, xlabel='Epoch'):
		if smooth: data1 = smoother(data1,2)
		fig, ax = plt.subplots(1, 1)
		ax.set_ylabel(name1)
		ax.plot(data1, linewidth=0.2)
		ax.set_xlabel(xlabel)
		fig.savefig(self.prefix2 + f'{name1}.pdf')
		plt.close()

	def plot2(self, name1, name2, data1, data2, smooth = True, xlabel='Epoch'):
		if smooth: data1, data2 = smoother(data1,2), smoother(data2,2)
		fig, ax = plt.subplots(1, 1, figsize=(3,1.9))
		ax.set_ylabel(name1); ax.set_xlabel(xlabel)
		l1 = ax.plot(data1, linewidth=0.6, label=name1, c = 'red')
		ax2 = ax.twinx()
		l2 = ax2.plot(data2, '--', linewidth=0.6, alpha=0.8, label=name2)
		ax2.set_xlabel(xlabel)
		ax2.set_ylabel(name2)
		plt.legend(handles=l1+l2, loc=9, bbox_to_anchor=(0.5, 1.2), ncol=2, prop={'size': 7})
		fig.savefig(self.prefix2 + f'{name1}_{name2}.pdf', pad_inches=0)
		plt.close()

	def plot_heatmap(self, title, name1, name2, data1, data2):
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 1.8))
		ax1.set_title(title)
		yticks = np.linspace(0, self.n_hosts, 4, dtype=np.int32)
		h1 = sns.heatmap(data1.transpose(),cmap="YlGnBu", yticklabels=yticks, linewidth=0.01, ax = ax1)
		h2 = sns.heatmap(data2.transpose(),cmap="YlGnBu", yticklabels=yticks, linewidth=0.01, ax = ax2)
		ax1.set_yticks(yticks); ax2.set_yticks(yticks); 
		xticks = np.linspace(0, data1.shape[0]-2, 5, dtype=np.int32)
		ax1.set_xticks(xticks); ax2.set_xticks(xticks); ax2.set_xticklabels(xticks, rotation=0)
		ax1.set_xticklabels(xticks, rotation=0)
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel(name2); ax1.set_ylabel(name1)
		fig.savefig(self.prefix2 + f'{title}_{name1}_{name2}.pdf', bbox_inches = 'tight')
		plt.close()

	def plot_tsne(self, name1, data1, labels1):
		fig, ax = plt.subplots(1, 1, figsize=(3, 2))
		target_ids = range(3); labs = ['CPU', 'RAM', 'Disk']
		for i, c, label in zip(target_ids, self.colors, labels1):
			ax.scatter(data1[labels1 == i, 0], data1[labels1 == i, 1], c=c, alpha=0.6, label=labs[i])
		ax.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.2))
		fig.savefig(self.prefix2 + f'tsne_{name1}.pdf', pad_inches=0)
		plt.close()

class GAN_Plotter():
	def __init__(self, env, gname, dname, training = True):
		self.env = env
		self.gname, self.dname = gname, dname
		self.n_hosts = int(gname.split('_')[-1])# 将gname按'_'分割，取最后一个元素转换成int类型的数字 16
		self.folder = os.path.join(plot_folder, env, 'gan' if training else 'test')
		self.prefix = self.folder + '/' + self.gname + '_' + self.dname
		self.epoch = 0
		self.improvement_ratio = []
		os.makedirs(self.folder, exist_ok=True)# 创建文件夹，避免报错
		for f in glob.glob(self.folder + '/*'): os.remove(f)# 若文件夹非空，删除文件夹下的所有文件
		plt.rcParams["font.family"] = "Times New Roman"
		self.init_params()

	def init_params(self):
		self.anomaly_detected = []
		self.class_detected = []
		self.hosts_migrated = []
		self.migrating = []
		self.new_score_better = []

	def update_anomaly_detected(self, detected):# 将参数detected(是1 /否0 检测到异常)添加到列表中
		self.anomaly_detected.append(detected)

	def update_class_detected(self, detected):# 将检测到的类别添加到列表中
		print(detected)
		self.class_detected.append(detected)

	def new_better(self, new_better):# new_better=1表示新决策更好，否则为0
		# self.new_score_better.append(new_better + 0)# +0用于将bool类型转换为int类型
		if not new_better: 
			self.hosts_migrated.append([0] * int(self.gname.split('_')[1]))
			self.migrating.append(0)
	# 新决策更好时调用
	def plot_test(self, hosts_from, epoch):
		self.migrating.append((np.sum(hosts_from) > 0) + 0)
		self.hosts_migrated.append(hosts_from)
		self.prefix2 = self.prefix + '_Test_' + str(epoch) + '_'
		# self.epoch += 1
		self.plot1('New Score Better', self.new_score_better)
		if epoch < 20: return
		self.plot_heatmap('Fault Prediction and Classification', 'Prediction', 'Classification', np.array(self.anomaly_detected).reshape(1, -1), np.array(self.class_detected))
		self.plot_heatmapc('Migrations', 'Migration', 'Hosts from Migration', np.array(self.migrating).reshape(1, -1), np.array(self.hosts_migrated))
		self.plot_heatmap_combined('Combined', np.array(self.anomaly_detected).reshape(1, -1), np.array(self.class_detected), np.array(self.migrating).reshape(1, -1), np.array(self.hosts_migrated))
	# 在线训练时调用
	def plot(self, accuracy_list, epoch, ns, os):
		self.prefix2 = self.prefix + '_' + str(epoch) + '_'
		# self.epoch += 1
		self.gloss_list = [i[0] for i in accuracy_list]
		self.dloss_list = [i[1] for i in accuracy_list]
		self.new_score_better.append((ns <= os) + 0)# 分数越低越好
		self.improvement_ratio.append(sum(self.new_score_better) / epoch)
		self.plot2('Generator Loss', 'Discriminator Loss', self.gloss_list, self.dloss_list)
		# self.plot3('Generator Loss', 'Discriminator Loss', 'Improvement Ratio', self.gloss_list, self.dloss_list, self.improvement_ratio)
		self.plot4( 'Improvement Ratio', self.improvement_ratio)
		self.plot1('New Score Better', self.new_score_better)
		if epoch < 20: return
		self.plot_heatmap('Fault Prediction and Classification', 'Prediction', 'Classification', np.array(self.anomaly_detected).reshape(1, -1), np.array(self.class_detected))

	def plot1(self, name1, data1, smooth = True, xlabel='Epoch'):
		if smooth: data1 = smoother(data1)
		fig, ax = plt.subplots(1, 1)
		ax.set_ylabel(name1)
		ax.plot(data1, linewidth=0.2)
		ax.set_xlabel(xlabel)
		fig.savefig(self.prefix2 + f'{name1}.pdf')
		plt.close()

	def plot2(self, name1, name2, data1, data2, smooth = True, xlabel='Iteration'):
		if smooth: data1, data2 = smoother(data1), smoother(data2, 2)
		fig, ax = plt.subplots(1, 1, figsize=(3,1.9))
		ax.set_ylabel(name1); ax.set_xlabel(xlabel)
		l1 = ax.plot(data1, linewidth=0.6, label=name1, c = 'red')
		ax2 = ax.twinx()
		l2 = ax2.plot(data2, '--', linewidth=0.6, alpha=0.8, label=name2)
		ax2.set_xlabel(xlabel)
		ax2.set_ylabel(name2)
		plt.legend(handles=l1+l2, loc=9, bbox_to_anchor=(0.5, 1.2), ncol=2, prop={'size': 7})
		fig.savefig(self.prefix2 + f'{name1}_{name2}.pdf', pad_inches=0)
		plt.close()

	def plot3(self, name1, name2, name3, data1, data2, data3, smooth = True, xlabel='Epoch'):
		if smooth: data1, data2, data3 = smoother(data1, 3), smoother(data2, 3), smoother(data3,2)
		fig, ax = plt.subplots(1, 1, figsize=(3,1.9))
		ax.set_ylabel(name1); ax.set_xlabel(xlabel)
		l1 = ax.plot(data1, linewidth=0.6, label=name1, c = 'red')
		ax2 = ax.twinx()
		l2 = ax2.plot(data2, '--', linewidth=0.6, alpha=0.8, label=name2)
		ax2.set_xlabel(xlabel)
		ax2.set_ylabel(name2)
		ax3 = ax.twinx()
		l3 = ax3.plot(data3, '.-', c = 'g', linewidth=0.6, alpha=0.6, label=name3)
		ax3.set_ylabel(name3); ax3.spines["right"].set_position(("axes", 1.25))
		plt.legend(handles=l1+l2+l3, loc=9, bbox_to_anchor=(0.5, 1.25), ncol=2, prop={'size': 7})
		fig.savefig(self.prefix2 + f'{name1}_{name2}_{name3}.pdf', pad_inches=0)
		plt.close()

	def plot4(self, name, data, smooth = True, xlabel='Epoch'):
		if smooth: data = smoother(data,2)
		fig, ax = plt.subplots(1, 1, figsize=(3,1.9))
		l = ax.plot(data, '.-', c = 'g', linewidth=0.6, alpha=0.6, label=name)
		ax.set_xlabel(xlabel); ax.set_ylabel(name); ax.spines["right"].set_position(("axes", 1.25))
		ax2 = ax.twinx()
		ax2.set_xlabel(xlabel)
		ax2.set_ylabel(name)
		l2 = ax2.plot(data, '.-', c = 'g', linewidth=0.6, alpha=0.6, label=name)
		plt.legend(handles=l, loc=9, bbox_to_anchor=(0.5, 1.25), ncol=2, prop={'size': 7})
		fig.savefig(self.prefix2 + f'{name}.pdf', pad_inches=0)
		plt.close()


	def plot_heatmap(self, title, name1, name2, data1, data2):
		from mpl_toolkits.axes_grid1 import make_axes_locatable

		# 创建图形和子图
		fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0.2, 1], 'hspace': 0.5}, figsize=(3.5, 1.8))
		ax1.set_title(title)
		yticks = np.linspace(0, self.n_hosts, 2, dtype=np.int32)

		# 从原始的YlGnBu色彩映射中获取两个颜色点
		from matplotlib.pyplot import cm
		ylgnbu = cm.get_cmap('YlGnBu')
		low_color = ylgnbu(0.99)  # 深蓝
		high_color = ylgnbu(0.01)  # 黄色
		# 创建两段式色彩映射
		two_tone_cmap = LinearSegmentedColormap.from_list('TwoToneYlGnBu',
														  [(0, low_color),
														   (0.5, low_color),
														   (0.501, high_color),
														   (1, high_color)],
														  N=256)

		dcmap = LinearSegmentedColormap.from_list('Custom', ['w', 'r', 'g', 'b'], 4)
		data2 = (data2 + 1).transpose()

		# 创建不带colorbar的热图
		h1 = sns.heatmap(data1, cmap=two_tone_cmap, yticklabels=[0], linewidths=0.02, linecolor='black', ax=ax1, cbar=False)
		h2 = sns.heatmap(data2, cmap=dcmap, yticklabels=yticks, linewidths=0.02, linecolor='black', ax=ax2, cbar=False)

		# 为colorbar创建自定义axes
		# 第一个colorbar axes
		divider1 = make_axes_locatable(ax1)
		cax1 = divider1.append_axes("right", size="3%", pad=0.12)

		# 第二个colorbar axes - 确保与第一个相同宽度和对齐
		divider2 = make_axes_locatable(ax2)
		cax2 = divider2.append_axes("right", size="3%", pad=0.12)

		# 创建ScalarMappables
		sm1 = plt.cm.ScalarMappable(cmap=two_tone_cmap, norm=plt.Normalize(vmin=0, vmax=1))
		sm2 = plt.cm.ScalarMappable(cmap=dcmap, norm=plt.Normalize(vmin=0, vmax=4))

		# 添加colorbar到自定义axes
		cbar1 = fig.colorbar(sm1, cax=cax1)
		cbar1.set_ticks([0.25, 0.75])
		cbar1.set_ticklabels(['0', '1'])

		cbar2 = fig.colorbar(sm2, cax=cax2)
		cbar2.set_ticks([0.5, 1.5, 2.5, 3.5])
		cbar2.set_ticklabels(['None', 'CPU', 'RAM', 'Disk'])

		# 设置其他图形属性
		ax1.set_yticks([0])
		ax2.set_yticks(yticks)
		ax2.set_yticklabels(yticks, rotation=0)
		xticks1 = np.linspace(0, data1.shape[1], 5, dtype=np.int32)
		xticks2 = np.linspace(0, data2.shape[1], 5, dtype=np.int32)
		ax1.set_xticks(xticks1)
		ax2.set_xticks(xticks2)
		ax2.set_xticklabels(xticks2, rotation=0)
		ax1.set_xticklabels(xticks1, rotation=0)
		ax1.set_xlabel('Interval', labelpad=1)
		ax2.set_xlabel('Epoch', labelpad=1)
		ax2.set_ylabel(name2, labelpad=-1)
		ax1.set_ylabel(name1, labelpad=-1)
		# 调整y轴标签位置，使其更靠近轴线
		ax1.yaxis.set_label_coords(-0.05, 0.5)
		ax2.yaxis.set_label_coords(-0.05, 0.5)
		fig.align_ylabels([ax1, ax2])

		# 为子图添加边框
		for spine in ax1.spines.values():
			spine.set_visible(True)
			spine.set_color('black')
			spine.set_linewidth(0.5)

		for spine in ax2.spines.values():
			spine.set_visible(True)
			spine.set_color('black')
			spine.set_linewidth(0.5)

		plt.tight_layout(pad=1.0)
		fig.savefig(self.prefix2 + f'{title}_{name1}_{name2}.pdf', bbox_inches='tight')
		plt.close()


	def plot_heatmapc(self, title, name1, name2, data1, data2):
		from mpl_toolkits.axes_grid1 import make_axes_locatable

		# 创建图形和子图
		fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0.2, 1], 'hspace': 0.5}, figsize=(3.5, 1.8))
		ax1.set_title(title)
		ax1.set_ylabel(name1)
		yticks = np.linspace(0, self.n_hosts, 2, dtype=np.int32)
		data2 = data2.transpose()

		from matplotlib.pyplot import cm
		ylgnbu = cm.get_cmap('YlGnBu')
		rdbur = cm.get_cmap('RdBu_r')

		low_color = ylgnbu(0.01)# 黄色
		high_color = rdbur(0.82)# 橙红

		# 创建两段式色彩映射
		two_tone_cmap = LinearSegmentedColormap.from_list('TwoToneYlGnBu',
														  [(0, low_color),
														   (0.5, low_color),
														   (0.501, high_color),
														   (1, high_color)],
														  N=256)

		# 创建不带colorbar的热图
		h1 = sns.heatmap(data1, cmap=two_tone_cmap, yticklabels=[0], linewidths=0.02, linecolor='black', ax=ax1, cbar=False)
		h2 = sns.heatmap(data2, cmap=two_tone_cmap, yticklabels=yticks, linewidths=0.02, linecolor='black', ax=ax2, cbar=False)

		# 为colorbar创建自定义axes
		# 第一个colorbar axes
		divider1 = make_axes_locatable(ax1)
		cax1 = divider1.append_axes("right", size="3%", pad=0.12)

		# 第二个colorbar axes - 确保与第一个相同宽度和对齐
		divider2 = make_axes_locatable(ax2)
		cax2 = divider2.append_axes("right", size="3%", pad=0.12)

		# 创建ScalarMappable
		sm = plt.cm.ScalarMappable(cmap=two_tone_cmap, norm=plt.Normalize(vmin=0, vmax=1))

		# 添加colorbar到自定义axes
		cbar1 = fig.colorbar(sm, cax=cax1)
		cbar1.set_ticks([0.25, 0.75])
		cbar1.set_ticklabels(['0', '1'])

		cbar2 = fig.colorbar(sm, cax=cax2)
		cbar2.set_ticks([0.25, 0.75])
		cbar2.set_ticklabels(['0', '1'])

		# 设置其他图形属性
		ax1.set_yticks([0]);
		ax2.set_yticks(yticks)
		ax2.set_yticklabels(yticks, rotation=0)
		xticks1 = np.linspace(0, data1.shape[1], 5, dtype=np.int32);
		xticks2 = np.linspace(0, data2.shape[1], 5, dtype=np.int32)
		ax1.set_xticks(xticks1)
		ax2.set_xticks(xticks2)
		ax2.set_xticklabels(xticks2, rotation=0)
		ax1.set_xticklabels(xticks1, rotation=0)
		ax2.set_xlabel('Epoch', labelpad=1)
		ax2.set_ylabel(name2, labelpad=-1)
		ax1.set_ylabel(name1, labelpad=-1)
		# 调整y轴标签位置，使其更靠近轴线
		ax1.yaxis.set_label_coords(-0.05, 0.5)
		ax2.yaxis.set_label_coords(-0.05, 0.5)
		fig.align_ylabels([ax1, ax2])

		# 为子图添加边框
		for spine in ax1.spines.values():
			spine.set_visible(True)
			spine.set_color('black')
			spine.set_linewidth(0.5)

		for spine in ax2.spines.values():
			spine.set_visible(True)
			spine.set_color('black')
			spine.set_linewidth(0.5)

		plt.tight_layout(pad=1.0)
		fig.savefig(self.prefix2 + f'{title}_{name1}_{name2}.pdf', bbox_inches='tight')
		plt.close()

	def plot_heatmap_combined(self, title, data1_anomaly, data2_class, data1_migration, data2_hosts):
		from mpl_toolkits.axes_grid1 import make_axes_locatable
		import matplotlib.gridspec as gridspec

		# Create figure with more space
		fig = plt.figure(figsize=(3.5, 6.5))

		# 创建外层GridSpec，调整第一个和第二个部分之间的间距
		outer_gs = gridspec.GridSpec(2, 1, height_ratios=[0.2, 1.8], hspace=0.15)

		# 创建内层GridSpec
		upper_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[0])
		
		# 对下半部分保持0.2的间距
		lower_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[1],
												height_ratios=[0.8, 0.2, 0.8], hspace=0.2)

		# Create four subplots
		ax1 = plt.subplot(upper_gs[0])  # First row - anomaly detection
		ax2 = plt.subplot(lower_gs[0])  # Second row - class prediction
		ax3 = plt.subplot(lower_gs[1])  # Third row - migration decision
		ax4 = plt.subplot(lower_gs[2])  # Fourth row - migration hosts

		# Rest of the code remains the same until labels...
		yticks = np.linspace(0, self.n_hosts, 2, dtype=np.int32)

		from matplotlib.pyplot import cm
		ylgnbu = cm.get_cmap('YlGnBu')
		rdbur = cm.get_cmap('RdBu_r')

		low_color = rdbur(0.05)# 深蓝
		mid_color = ylgnbu(0.01)# 黄色
		high_color = rdbur(0.82)# 橙红

		two_tone_cmap1 = LinearSegmentedColormap.from_list('TwoToneYlGnBu',
														  [(0, low_color),
														   (0.5, low_color),
														   (0.501, mid_color),
														   (1, mid_color)],
														  N=256)
		two_tone_cmap2 = LinearSegmentedColormap.from_list('TwoToneYlGnBu',
														  [(0, mid_color),
														   (0.5, mid_color),
														   (0.501, high_color),
														   (1, high_color)],
														  N=256)
		dcmap = LinearSegmentedColormap.from_list('Custom', ['w', 'r', 'g', 'b'], 4)

		data2_class = (data2_class + 1).transpose()
		data2_hosts = data2_hosts.transpose()

		h1 = sns.heatmap(data1_anomaly, cmap=two_tone_cmap1, yticklabels=[0], linewidths=0.02, linecolor='black', ax=ax1,
						 cbar=False)
		h2 = sns.heatmap(data2_class, cmap=dcmap, yticklabels=yticks, linewidths=0.02, linecolor='black', ax=ax2,
						 cbar=False)
		h3 = sns.heatmap(data1_migration, cmap=two_tone_cmap2, yticklabels=[0], linewidths=0.02, linecolor='black',
						 ax=ax3, cbar=False)
		h4 = sns.heatmap(data2_hosts, cmap=two_tone_cmap2, yticklabels=yticks, linewidths=0.02, linecolor='black',
						 ax=ax4, cbar=False)

		# 创建colorbar
		for ax, subplot_cmap, vmax, tickpos, ticklabels in [
			(ax1, two_tone_cmap1, 1, [0.25, 0.75], ['0', '1']),
			(ax2, dcmap, 4, [0.5, 1.5, 2.5, 3.5], ['None', 'CPU', 'RAM', 'Disk']),
			(ax3, two_tone_cmap2, 1, [0.25, 0.75], ['0', '1']),
			(ax4, two_tone_cmap2, 1, [0.25, 0.75], ['0', '1'])
		]:
			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="3%", pad=0.08)
			sm = plt.cm.ScalarMappable(cmap=subplot_cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
			cbar = fig.colorbar(sm, cax=cax)
			cbar.set_ticks(tickpos)
			cbar.set_ticklabels(ticklabels)

		# 设置刻度和标签
		ax1.set_yticks([0])
		xticks1 = np.linspace(0, data1_anomaly.shape[1], 5, dtype=np.int32)
		ax1.set_xticks(xticks1)
		ax1.set_xticklabels(xticks1, rotation=0)
		# Set xlabel position explicitly to avoid overlap
		ax1.set_xlabel('Interval')
		ax1.xaxis.set_label_coords(0.5, -0.4)  # Adjust this value as needed

		ax2.set_yticks(yticks)
		ax2.set_yticklabels(yticks, rotation=0)
		xticks2 = np.linspace(0, data2_class.shape[1], 5, dtype=np.int32)
		ax2.set_xticks(xticks2)
		ax2.set_xticklabels(xticks2, rotation=0)
		ax2.set_xlabel('')  # 第二个子图不显示xlabel

		ax3.set_yticks([0])
		xticks3 = np.linspace(0, data1_migration.shape[1], 5, dtype=np.int32)
		ax3.set_xticks(xticks3)
		ax3.set_xticklabels(xticks3, rotation=0)
		ax3.set_xlabel('')

		ax4.set_yticks(yticks)
		ax4.set_yticklabels(yticks, rotation=0)
		xticks4 = np.linspace(0, data2_hosts.shape[1], 5, dtype=np.int32)
		ax4.set_xticks(xticks4)
		ax4.set_xticklabels(xticks4, rotation=0)
		ax4.set_xlabel('Epoch', labelpad=1)

		# 设置y轴标签
		ax1.set_ylabel('Prediction', labelpad=-1)
		ax2.set_ylabel('Fault Class for each Host', labelpad=-1)
		ax3.set_ylabel('Migration', labelpad=-1)
		ax4.set_ylabel('Hosts from Migration', labelpad=-1)

		# 调整y轴标签位置
		for ax in [ax1, ax2, ax3, ax4]:
			ax.yaxis.set_label_coords(-0.03, 0.5)

		# 添加边框
		for ax in [ax1, ax2, ax3, ax4]:
			for spine in ax.spines.values():
				spine.set_visible(True)
				spine.set_color('black')
				spine.set_linewidth(0.5)

		plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.95])
		fig.savefig(self.prefix2 + f'{title}_Combined.pdf', bbox_inches='tight')
		plt.close()