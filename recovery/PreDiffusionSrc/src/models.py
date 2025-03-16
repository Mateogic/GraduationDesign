import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from .constants import *
from .dlutils import *

## FPE
class FPE_16(nn.Module):
	def __init__(self):
		super(FPE_16, self).__init__()
		self.name = 'FPE_16'
		self.lr = 0.0001
		self.n_hosts = 16
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		self.gru = nn.GRU(self.n_window, self.n_window, 1)
		src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
		self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
		self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		self.anomaly_decoder = nn.Sequential(
			nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
		)
		self.prototype_decoder = nn.Sequential(
			nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		h = torch.randn(1, self.n_window, dtype=torch.double)
		gru_t, _ = self.gru(torch.t(t), h)
		gru_t = torch.t(gru_t)
		graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
		gat_t = self.gat(torch.t(graph))
		gat_t = torch.t(gat_t)
		concat_t = torch.cat((gru_t, gat_t), dim=1)
		o, _ = self.mha(concat_t, concat_t, concat_t)
		t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)	
		return t

	def anomaly_decode(self, t):
		anomaly_scores = []
		for elem in t:
			anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))	
		return anomaly_scores

	def prototype_decode(self, t):
		prototypes = []
		for elem in t:
			prototypes.append(self.prototype_decoder(elem))	
		return prototypes

	def forward(self, t, s):
		t = self.encode(t, s)
		anomaly_scores = self.anomaly_decode(t)
		prototypes = self.prototype_decode(t)
		return anomaly_scores, prototypes

# Generator Network : Input = Schedule(S), Embedding(E^F); Output = New Schedule(N)
class Gen_16(nn.Module):
	def __init__(self):
		super(Gen_16, self).__init__()
		self.name = 'Gen_16'
		self.lr = 0.00005# 学习率
		self.n_hosts = 16# 主机数量
		self.n_hidden = 64# 隐藏层神经元数量
		self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts# 输入维度
		self.delta = nn.Sequential(# 定义顺序容器
			nn.Linear(self.n, self.n_hidden),# 全连接层，将输入维度映射到隐藏层维度
			nn.LeakyReLU(True),# 使用LeakyReLu激活函数
			nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts),# 全连接层，将隐藏层维度映射到输出维度
			nn.Tanh(),# Tanh函数
		)

	def forward(self, e, s):
		# 将嵌入e和调度s展平连接，输入到delta顺序容器(Gen网络)，输出新的调度数据增量
		del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
		# 将增量delta加到原始调度s上，返回新的调度n
		return s + del_s.reshape(self.n_hosts, self.n_hosts)

# Discriminator Network : Input = Schedule, New Schedule; Output = Likelihood scores
class Disc_16(nn.Module):
	def __init__(self):
		super(Disc_16, self).__init__()
		self.name = 'Disc_16'
		self.lr = 0.00005
		self.n_hosts = 16
		self.n_hidden = 64
		self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
		self.probs = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
		)

	def forward(self, o, n):
		probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
		return probs


## FPE
class FPE_50(nn.Module):
	def __init__(self):
		super(FPE_50, self).__init__()
		self.name = 'FPE_50'
		self.lr = 0.0001
		self.n_hosts = 50
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 50
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		self.gru = nn.GRU(self.n_window, self.n_window, 1)
		src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
		self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
		self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		self.anomaly_decoder = nn.Sequential(
			nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
		)
		self.prototype_decoder = nn.Sequential(
			nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		h = torch.randn(1, self.n_window, dtype=torch.double)
		gru_t, _ = self.gru(torch.t(t), h)
		gru_t = torch.t(gru_t)
		graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
		gat_t = self.gat(torch.t(graph))
		gat_t = torch.t(gat_t)
		concat_t = torch.cat((gru_t, gat_t), dim=1)
		o, _ = self.mha(concat_t, concat_t, concat_t)
		t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)	
		return t

	def anomaly_decode(self, t):
		anomaly_scores = []
		for elem in t:
			anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))	
		return anomaly_scores

	def prototype_decode(self, t):
		prototypes = []
		for elem in t:
			prototypes.append(self.prototype_decoder(elem))	
		return prototypes

	def forward(self, t, s):
		t = self.encode(t, s)
		anomaly_scores = self.anomaly_decode(t)
		prototypes = self.prototype_decode(t)
		return anomaly_scores, prototypes

## Simple Multi-Head Self-Attention Model
class Attention_50(nn.Module):
	def __init__(self):
		super(Attention_50, self).__init__()
		self.name = 'Attention_50'
		self.lr = 0.0008
		self.n_hosts = 50
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		# self.atts = [ nn.Sequential( nn.Linear(self.n, self.n_feats * self.n_feats), 
		# 		nn.Sigmoid())	for i in range(1)]
		# self.encoder_atts = nn.ModuleList(self.atts)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * self.n_feats, self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		self.anomaly_decoder = nn.Sequential(
			nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
		)
		self.prototype_decoder = nn.Sequential(
			nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		# for at in self.encoder_atts:
		# 	inp = torch.cat((t.view(-1), s.view(-1)))
		# 	ats = at(inp).reshape(self.n_feats, self.n_feats)
		# 	t = torch.matmul(t, ats)	
		t = self.encoder(t.view(-1)).view(self.n_hosts, self.n_latent)	
		return t

	def anomaly_decode(self, t):
		anomaly_scores = []
		for elem in t:
			anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))	
		return anomaly_scores

	def prototype_decode(self, t):
		prototypes = []
		for elem in t:
			prototypes.append(self.prototype_decoder(elem))	
		return prototypes

	def forward(self, t, s):
		t = self.encode(t, s)
		anomaly_scores = self.anomaly_decode(t)
		prototypes = self.prototype_decode(t)
		return anomaly_scores, prototypes

# Generator Network : Input = Schedule, Embedding; Output = New Schedule
class Gen_50(nn.Module):
	def __init__(self):
		super(Gen_50, self).__init__()
		self.name = 'Gen_50'
		self.lr = 0.00003
		self.n_hosts = 50
		self.n_hidden = 64
		self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
		self.delta = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
		)

	def forward(self, e, s):
		del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
		return s + del_s.reshape(self.n_hosts, self.n_hosts)

# Discriminator Network : Input = Schedule, New Schedule; Output = Likelihood scores
class Disc_50(nn.Module):
	def __init__(self):
		super(Disc_50, self).__init__()
		self.name = 'Disc_50'
		self.lr = 0.00003
		self.n_hosts = 50
		self.n_hidden = 64
		self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
		self.probs = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
		)

	def forward(self, o, n):
		probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
		return probs


############## PreGANPro Models ##############

# TransformerPro Model
class TransformerPro_16(nn.Module):
	def __init__(self):
		super(TransformerPro_16, self).__init__()
		self.name = 'TransformerPro_16'
		self.lr = 0.0001
		self.n_hosts = 16
		feats = 3 * self.n_hosts
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.multi = 2# 拓展倍数
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
		self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
		
		# 修改这里 - 使用正确的嵌入维度
		# 对于GAT输出，实际维度是49而不是self.n_window(3)
		self.embedding_dim = self.n_feats + 1  # 这是GAT输出的实际维度，为49(48+1)，其中的1是额外的全局节点，用于捕获关系
		# self.hist_attention = nn.MultiheadAttention(self.embedding_dim, num_heads=1, dtype=torch.double)
		self.hist_attention = nn.MultiheadAttention(self.embedding_dim, num_heads=1, dtype=torch.double)
		# 对应修改融合层的输入和输出维度
		self.fusion_layer = nn.Linear(self.embedding_dim * 2, self.embedding_dim).double()

		self.time_encoder = nn.Sequential(
			nn.Linear(feats, feats * 2 + 1), 
		)
		self.pos_encoder = PositionalEncoding(feats * 2 + 1, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats * self.multi + 1, nhead=1, dropout=0.1)
		self.encoder = TransformerEncoder(encoder_layers, 1)
		a_decoder_layers = TransformerDecoderLayer(d_model=feats * self.multi + 1, nhead=1, dropout=0.1)
		self.anomaly_decoder = TransformerDecoder(a_decoder_layers, 1)
		self.anomaly_decoder2 = nn.Sequential(
			nn.Linear((feats * 2 + 1) * self.n_window * self.n_window, 2 * self.n_hosts), 
		)
		self.softm = nn.Softmax(dim=1)
		p_decoder_layers = TransformerDecoderLayer(d_model=feats * self.multi + 1, nhead=1, dropout=0.1)
		self.prototype_decoder = TransformerDecoder(p_decoder_layers, 1)
		self.prototype_decoder2 = nn.Sequential(
			nn.Linear((feats * 2 + 1) * self.n_window * self.n_window, PROTO_DIM * self.n_hosts), 
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s, historical_embeddings=None):
		t = torch.squeeze(t, 1)
		graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
		gat_t = self.gat(torch.t(graph))
		gat_t = torch.t(gat_t)
		
		# 保存当前GAT输出用于返回
		current_gat = gat_t.clone()
		
		# 如果有历史嵌入，连接历史嵌入和当前GAT结果
		if historical_embeddings and len(historical_embeddings) > 0:
			# 将历史嵌入转换为张量，保持double类型
			hist_tensor = torch.stack(list(historical_embeddings))
			
			# 使用预定义的注意力层，无需类型转换
			gat_t_batch = gat_t.unsqueeze(0)
			hist_context, _ = self.hist_attention(gat_t_batch, 
												hist_tensor, 
												hist_tensor)
			
			# 无需类型转换，直接获取结果
			hist_context = hist_context.squeeze(0)
			
			# 连接历史上下文和当前GAT输出
			enhanced_gat = torch.cat([gat_t, hist_context], dim=1)
			
			# 使用预定义的融合层
			gat_t = self.fusion_layer(enhanced_gat)
		
		o = torch.cat((t, gat_t), dim=1)
		t = o * math.sqrt(self.n_feats)
		t = self.pos_encoder(t)
		memory = self.encoder(t)    
		return memory, current_gat

	def anomaly_decode(self, t, memory):
		anomaly_scores = self.anomaly_decoder(t, memory)
		anomaly_scores = self.anomaly_decoder2(anomaly_scores.view(-1)).view(-1, 1, 2)
		return anomaly_scores

	def prototype_decode(self, t, memory):
		prototypes = self.prototype_decoder(t, memory)
		prototypes = self.prototype_decoder2(prototypes.view(-1)).view(-1, PROTO_DIM)
		return prototypes

	def forward(self, t, s, historical_embeddings=None, return_embedding=False):
		encoded_t = self.time_encoder(t).unsqueeze(dim=1).expand(-1, self.n_window, -1)
		t = t.unsqueeze(dim=1)
		memory, current_gat = self.encode(t, s, historical_embeddings)
		anomaly_scores = self.anomaly_decode(encoded_t, memory)
		prototypes = self.prototype_decode(encoded_t, memory)
		
		if return_embedding:
			return anomaly_scores, prototypes, current_gat
		return anomaly_scores, prototypes

############## Diffusion_16 Models ##############

# Diffusion_16 Model
class Diffusion_16(nn.Module):
	def __init__(self):
		super(Diffusion_16, self).__init__()
		self.name = 'Diffusion_16'
		self.lr = 0.00005
		self.n_hosts = 16
		self.n_hidden = 64
		self.proto_dim = PROTO_DIM  # 从常量导入
		
		# 计算输入维度 - 确保与数据集匹配
		self.n = self.n_hosts * self.proto_dim + self.n_hosts * self.n_hosts
		input_dim = self.n_hosts * self.n_hosts + (self.n_hidden // 2 + 1) + self.n_hidden
		
		# PNDM相关参数
		self.n_steps = 20  # 扩散步数，训练时使用
		self.n_actual_steps = 10  # 实际推理时使用的步数
		self.beta_start = 0.0001
		self.beta_end = 0.02
		
		# 添加条件处理网络
		self.condition_encoder = nn.Sequential(
			nn.Linear(self.n, self.n_hidden),
			nn.SiLU(),
			nn.LayerNorm(self.n_hidden),
			nn.Linear(self.n_hidden, self.n_hidden),
		).double()

		# 注册beta schedule和相关参数为缓冲区(不是模型参数)
		self.register_buffer('betas', self._get_beta_schedule())
		alphas = 1.0 - self.betas
		self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
		
		# 降噪网络 - 增强表示能力
		self.denoiser = nn.Sequential(
			nn.Linear(input_dim, self.n_hidden * 2),
			nn.SiLU(),
			nn.LayerNorm(self.n_hidden * 2),  # 添加层归一化
			nn.Linear(self.n_hidden * 2, self.n_hidden * 2),
			nn.SiLU(),
			nn.LayerNorm(self.n_hidden * 2),
			nn.Linear(self.n_hidden * 2, self.n_hidden * 2),
			nn.SiLU(),
			nn.Linear(self.n_hidden * 2, self.n_hosts * self.n_hosts)
		).double()

	def _get_beta_schedule(self):
		"""余弦Beta调度，比线性调度产生更高质量的样本"""
		steps = self.n_steps
		s = 0.008  # 控制最小和最大噪声水平
		
		# 余弦调度公式实现
		x = torch.linspace(0, steps, steps + 1)
		alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
		alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
		betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
		
		# 限制beta范围，提高数值稳定性
		betas = torch.clip(betas, 0.0001, 0.02)
		return betas.double()
	
	def forward(self, e, s, env_stats=None, run_simulation_fn=None):
		"""改进的前向推理，添加后处理优化"""
		with torch.no_grad():
			# 生成多个候选方案并选择最佳的
			best_delta = None
			best_score = float('inf')
			
			# 生成候选方案 - 无评估时只生成一个
			candidates_count = 1 if env_stats is None or run_simulation_fn is None else 3
			
			for _ in range(candidates_count):
				delta = self._pndm_sampling(e, s, steps=self.n_actual_steps)
				candidate = s + 4 * torch.tanh(delta.reshape(self.n_hosts, self.n_hosts))
				
				# 只在提供评估函数时评估
				if env_stats is not None and run_simulation_fn is not None:
					score = run_simulation_fn(env_stats, candidate)
					
					if score < best_score:
						best_score = score
						best_delta = delta
				else:
					# 无评估函数时直接使用第一个结果
					best_delta = delta
			
			# 返回最佳结果
			return s + 4 * torch.tanh(best_delta.reshape(self.n_hosts, self.n_hosts))
		
	def _pndm_sampling(self, e, s, steps=10):
		"""优化的PNDM采样方法，使用非均匀时间步和动态步长控制"""
		# 初始化噪声
		x = torch.randn(self.n_hosts * self.n_hosts).double()
		
		# 准备条件信息 - 确保维度匹配
		e_flat = e.reshape(-1)
		s_flat = s.reshape(-1)
		if len(e_flat) != self.n_hosts * self.proto_dim:
			# 如果嵌入维度不匹配，调整到正确大小
			if len(e_flat) > self.n_hosts * self.proto_dim:
				e_flat = e_flat[:self.n_hosts * self.proto_dim]
			else:
				padding = torch.zeros(self.n_hosts * self.proto_dim - len(e_flat)).double()
				e_flat = torch.cat([e_flat, padding])
		
		cond = torch.cat((e_flat, s_flat))
		
		# PNDM采样核心逻辑 - 使用线性多步方法加速
		phi_1 = x.clone()
		phi_2 = x.clone()
		phi_3 = x.clone()
		
		# 非均匀时间步，关注中低噪声区域
		# 使用指数缩放获取更密集的后期步长
		time_scales = torch.linspace(0, 1, steps + 1)[:-1] ** 1.5  # 指数缩放参数(>1更关注后期)
		time_steps = 1.0 - time_scales  # 从1倒数到接近0
		
		# 动态步长优化参数
		momentum_strength = 0.12  # 动量强度
		late_stage_boost = 1.25   # 后期阶段步长增强因子
		
		# 4阶线性多步PNDM - 优化版本
		with torch.no_grad():
			for i, t in enumerate(time_steps):
				# 计算当前时间步对应的索引，基于实际beta调度
				index = int(t * (self.n_steps - 1))
				alpha_cumprod_t = self.alphas_cumprod[index]
				
				# 当前时间步的噪声预测
				x_input = x / torch.sqrt(alpha_cumprod_t)
				t_input = torch.tensor([t]).double()
				noise_pred = self._denoise(x_input, t_input, cond)
				
				# 确定是否处于后期阶段
				is_late_stage = i >= int(steps * 0.7)
				
				# 动态步长增强因子和动量项
				if is_late_stage:
					# 后期阶段使用更大的步长
					step_boost = late_stage_boost
					# 添加动量项 - 向前一步方向推动
					if i > 0:
						momentum = momentum_strength * (x - phi_1)
					else:
						momentum = torch.zeros_like(x)
				else:
					step_boost = 1.0
					momentum = torch.zeros_like(x)
				
				# 根据阶段应用不同的数值方法
				if i == 0:
					# 第一步使用增强的欧拉方法
					x = self._euler_step(x, noise_pred, t, index, step_boost)
					phi_1 = x.clone()
				elif i == 1:
					# 第二步使用改进的欧拉方法
					x = self._midpoint_step(x, noise_pred, phi_1, t, index, step_boost)
					phi_2 = x.clone()
				elif i == 2:
					# 第三步使用优化的RK4方法
					x = self._rk4_step(x, noise_pred, phi_1, phi_2, t, index, step_boost)
					phi_3 = x.clone()
				else:
					# 后续步骤使用优化的4阶Adams-Bashforth
					x = self._adams_bashforth_step(x, noise_pred, phi_1, phi_2, phi_3, t, index, step_boost)
					
					# 应用动量项 - 修正布尔判断，使用torch.norm()检查是否有动量
					momentum_magnitude = torch.norm(momentum)
					if momentum_magnitude > 1e-8:  # 使用小阈值代替精确的零检查
						x = x - momentum
					
					# 更新历史状态
					phi_3 = phi_2.clone()
					phi_2 = phi_1.clone()
					phi_1 = x.clone()
				
				# 后期阶段优化：引导向相似步长变化方向
				if is_late_stage and i > 1:
					# 计算当前增量和前一步增量
					curr_delta = (x - phi_1)
					prev_delta = (phi_1 - phi_2)
					
					# 检查方向一致性 - 使用点积而非直接比较
					consistency = torch.sum(curr_delta * prev_delta)
					if consistency > 0:  # 方向一致时增强
						consistency_boost = 0.08
						x = x + consistency_boost * prev_delta
		
		return x.reshape(self.n_hosts, self.n_hosts)
		
	def _denoise(self, x_t, t, cond):
		"""增强版降噪函数，使用条件编码器"""
		try:
			# 使用条件编码器处理条件信息
			encoded_cond = self.condition_encoder(cond)
			
			# 时间嵌入改进 - 使用正弦位置编码
			t_scaled = t.item() * 1000  # 将t缩放到更大范围
			freqs = torch.exp(torch.linspace(0, 8, self.n_hidden // 4).double() * -math.log(10000.0))
			args = t_scaled * freqs
			t_embedding = torch.cat([torch.sin(args), torch.cos(args), t.view(-1)]).double()
			
			# 组合噪声数据和增强的条件信息
			input_tensor = torch.cat([x_t.view(-1), t_embedding, encoded_cond])
			
			# 预测噪声
			noise_pred = self.denoiser(input_tensor)
			return noise_pred
		except RuntimeError as e:
			# 错误处理...
			raise
		
	# 各种数值积分方法的实现
	def _euler_step(self, x, noise_pred, t, index, boost=1.0):
		"""带增强因子的欧拉步骤"""
		step_size = self.betas[index] / torch.sqrt(1 - self.alphas_cumprod[index])
		return x - step_size * boost * noise_pred

	def _midpoint_step(self, x, noise_pred, phi_1, t, index, boost=1.0):
		"""改进欧拉方法实现，支持步长增强"""
		step_size = self.betas[index] / torch.sqrt(1 - self.alphas_cumprod[index])
		return phi_1 - step_size * boost * noise_pred

	def _rk4_step(self, x, noise_pred, phi_1, phi_2, t, index, boost=1.0):
		"""RK4类似步骤实现，支持步长增强"""
		step_size = self.betas[index] / torch.sqrt(1 - self.alphas_cumprod[index])
		return phi_2 - step_size * boost * noise_pred

	def _adams_bashforth_step(self, x, noise_pred, phi_1, phi_2, phi_3, t, index, boost=1.0):
		"""Adams-Bashforth步骤实现(4阶线性多步法)，支持步长增强"""
		step_size = self.betas[index] / torch.sqrt(1 - self.alphas_cumprod[index])
		return phi_3 - step_size * boost * noise_pred