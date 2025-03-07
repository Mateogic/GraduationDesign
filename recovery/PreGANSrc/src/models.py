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


############## PreGANPlus Models ##############

# Transformer Model
class Transformer_16(nn.Module):
	def __init__(self):
		super(Transformer_16, self).__init__()
		self.name = 'Transformer_16'
		self.lr = 0.0001
		self.n_hosts = 16
		feats = 3 * self.n_hosts
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
		self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
		
		# 预定义历史嵌入处理所需的层，避免重复创建
		self.hist_attention = nn.MultiheadAttention(self.n_window, num_heads=1, dtype=torch.double)
		# 预定义融合层，使用double类型
		self.fusion_layer = nn.Linear(self.n_window * 2, self.n_window).double()

		self.time_encoder = nn.Sequential(
			nn.Linear(feats, feats * 2 + 1), 
		)
		self.pos_encoder = PositionalEncoding(feats * 2 + 1, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
		self.encoder = TransformerEncoder(encoder_layers, 1)
		a_decoder_layers = TransformerDecoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
		self.anomaly_decoder = TransformerDecoder(a_decoder_layers, 1)
		self.anomaly_decoder2 = nn.Sequential(
			nn.Linear((feats * 2 + 1) * self.n_window * self.n_window, 2 * self.n_hosts), 
		)
		self.softm = nn.Softmax(dim=1)
		p_decoder_layers = TransformerDecoderLayer(d_model=feats * 2 + 1, nhead=1, dropout=0.1)
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