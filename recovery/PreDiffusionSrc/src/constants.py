# Directory paths
model_folder = 'recovery/PreDiffusionSrc/checkpoints/prediffusion/'
data_folder = 'recovery/PreDiffusionSrc/data/'
plot_folder = 'recovery/PreDiffusionSrc/plots'# 绘图结果
data_filename = 'time_series.npy'# 离线训练FPE模型时 和 在线微调FPE时加载
schedule_filename = 'schedule_series.npy'# 离线训练FPE模型时加载

# Hyperparameters
num_epochs = 100# 训练轮数
PERCENTILES = 98
PROTO_DIM = 2# 原型向量的维度
PROTO_UPDATE_FACTOR = 0.2# 更新因子的初始值\alpha
PROTO_UPDATE_MIN = 0.01# 更新因子的最小值
PROTO_FACTOR_DECAY = 0.995# 更新因子的衰减系数1-\epsilon
# PROTO_FACTOR_DECAY = 0.995# 更新因子的衰减系数1-\epsilon
LATEST_WINDOW_SIZE = 10# 滑动窗口长度k
# LATEST_WINDOW_SIZE = 10# 滑动窗口长度k

# GAN parameters
# 源代码是0.8 0.2 此处根据论文修改为0.5 0.5
Coeff_Energy = 0.8# 平均能耗的加权系数\beta
Coeff_Latency = 0.2# 平均延迟的加权系数1-\beta