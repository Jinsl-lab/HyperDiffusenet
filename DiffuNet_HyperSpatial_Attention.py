import torch.nn as nn
# 0.341 15108 ;pca 0.314
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import NBLoss, ZINBLoss, MeanAct, DispAct, PoissonLoss
from tsne_helper import compute_gaussian_perplexity
from lorentzian_helper import *
import numpy as np
import math, os
from sklearn.metrics import pairwise_distances
from wrapped_normal import HyperbolicWrappedNorm

np.random.seed(42)
eps = 1e-6
weight_decay = 1e-3
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

torch.autograd.set_detect_anomaly(True)


def buildNetwork(layers, type, activation="elu", prob=0.):
    net = []
    for i in range(1, len(layers)):
        # 创建线性层并初始化
        layer = nn.Linear(layers[i - 1], layers[i])
        if activation == 'relu':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        else:
            nn.init.xavier_normal_(layer.weight)

        # 添加线性层到网络列表
        net.append(layer)

        # 添加批处理归一化层
        net.append(nn.BatchNorm1d(layers[i]))

        # 根据激活函数类型添加激活层
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif activation == "elu":
            net.append(nn.ELU())

        # 如果指定，添加Dropout层
        if prob > 0:
            net.append(nn.Dropout(p=prob))

    return nn.Sequential(*net)

class DiffusionToHyperbolic(nn.Module):
    """网络层用于将扩散映射的输出转换为双曲空间坐标"""

    def __init__(self, input_dim, output_dim):
        super(DiffusionToHyperbolic, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()  # 使用双曲正切作为非线性激活函数

    def forward(self, x):
        x = self.linear(x)
        return self.tanh(x)

class GraphTransformationLayer(nn.Module):
    def __init__(self, in_features, num_relations):
        super(GraphTransformationLayer, self).__init__()
        # in_features 是邻接矩阵的尺寸，即 512
        self.num_relations = num_relations
        self.transformations = nn.ParameterList([
            nn.Parameter(torch.randn(in_features, in_features))  # 修改为 in_features x in_features
            for _ in range(num_relations)
        ])

    def forward(self, adjacencies):
        # 如果adjacencies是一个单独的矩阵而不是列表，则需要将其封装成列表
        if isinstance(adjacencies, torch.Tensor):
            adjacencies = [adjacencies for _ in range(self.num_relations)]

        new_adj = sum(
            [F.relu(torch.matmul(adj, trans))
             for adj, trans in zip(adjacencies, self.transformations)])
        return new_adj


class GraphDiffusionConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dropout_rate=0.5):
        super(GraphDiffusionConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # 添加批量归一化层
        self.bn = nn.BatchNorm1d(out_features).to('cuda')
        self.dropout = nn.Dropout(dropout_rate)  # 添加 Dropout 层

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight.to(input.device))
        output = torch.spmm(adj, support)  # Make sure adj is a sparse tensor if using torch.spmm
        if self.bias is not None:
            output = output + self.bias.to(input.device)
        # 应用批量归一化
        output = self.bn(output)
        output = self.dropout(F.leaky_relu(output, negative_slope=0.01))  # 使用 LeakyReLU 激活和 Dropout
        # return F.leaky_relu(output, 0.01)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class CrossAttentionSpatialWeight(nn.Module):
    def __init__(self, gene_dim, spatial_dim, hidden_dim, output_dim):
        super(CrossAttentionSpatialWeight, self).__init__()
        self.query = nn.Linear(gene_dim, hidden_dim)
        self.key = nn.Linear(gene_dim, hidden_dim)
        self.value = nn.Linear(gene_dim, output_dim)
        self.spatial_transform = nn.Linear(spatial_dim, hidden_dim)
        # self.advanced_spatial_layer = nn.Linear(hidden_dim, hidden_dim)
        self.advanced_spatial_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # 添加非线性激活函数
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()  # 另一层非线性以增加处理能力
        )
        self.final_spatial_projection = nn.Linear(hidden_dim, 1)  # 投影到1维以便生成(N, N)矩阵
        self.softmax = nn.Softmax(dim=-1)
        self.dk = hidden_dim ** 0.5

    def forward(self, gene_expr, spatial_pos):
        Q = self.query(gene_expr)
        K = self.key(gene_expr)
        V = self.value(gene_expr)

        spatial_info = self.spatial_transform(spatial_pos)
        enhanced_spatial_info = self.advanced_spatial_layer(spatial_info)
        final_spatial_info = self.final_spatial_projection(enhanced_spatial_info).squeeze(-1)

        # 使用广播生成差值矩阵，并保证数值稳定性
        final_spatial_info = final_spatial_info.unsqueeze(1) - final_spatial_info.unsqueeze(0)
        final_spatial_info = torch.tanh(final_spatial_info)  # 使用 tanh 来限制输出范围，增加稳定性

        spatial_similarity = self.compute_spatial_weight(spatial_pos)
        expression_similarity = self.compute_cosine_similarity(gene_expr)
        combined_similarity = spatial_similarity + expression_similarity

        attention_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.dk
        attention_logits += combined_similarity
        attention_logits += final_spatial_info

        # 在应用 softmax 前减去最大值以提高数值稳定性
        # attention_logits = attention_logits - torch.max(attention_logits, dim=1, keepdim=True)[0]
        # attention_weights = self.softmax(attention_logits)
        # 对每个样本的logits减去其最大值以增加数值稳定性
        max_logits = torch.max(attention_logits, dim=1, keepdim=True)[0]
        attention_logits = attention_logits - max_logits
        attention_weights = self.softmax(attention_logits)

        updated_gene_expr = torch.matmul(attention_weights, V)
        return attention_weights, updated_gene_expr

    @staticmethod
    def compute_cosine_similarity(x):
        normalized_x = F.normalize(x, p=2, dim=1)
        cosine_similarity = torch.mm(normalized_x, normalized_x.t())
        return cosine_similarity

    @staticmethod
    def compute_spatial_weight(s, sigma=1.0):
        dist_s = torch.cdist(s, s, p=2)
        w_ij = torch.exp(-dist_s ** 2 / (2 * sigma ** 2))
        return w_ij


class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        self.loss_min = loss


###引入扩散模型来优化双曲空间嵌入表示
def compute_cosine_similarity(x):
    """
        计算余弦相似性矩阵。
        参数:
        - x: 基因表达数据，形状为 (n_samples, n_features)
    """
    normalized_x = F.normalize(x, p=2, dim=1)
    cosine_similarity = torch.mm(normalized_x, normalized_x.t())
    return cosine_similarity


# 定义空间相似性权重计算函数
def compute_spatial_weight(s, sigma=1.0):
    """
    计算基于空间位置的相似性权重
    Args:
        s: 空间位置矩阵，尺寸为 [batch_size, 2]
        sigma: 高斯核的带宽参数

    Returns:
        w_ij: 空间相似性权重矩阵
    """
    dist_s = torch.cdist(s, s, p=2)  # 计算空间位置之间的欧几里得距离
    w_ij = torch.exp(-dist_s ** 2 / (2 * sigma ** 2))  # 计算相似性权重
    return w_ij


class HyperDiffuseNet(nn.Module):
    def __init__(self, X, S, input_dim, encodeLayer=[], decodeLayer=[], batch_size=512,
                 activation="relu", z_dim=2, alpha=1., beta=1., gamma=1., perplexity=[30.],
                 prob=0., likelihood_type="nb", device="cuda"):
        super(HyperDiffuseNet, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.activation = activation
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        #self.gamma = gamma  # If gamma = 1, the Cauchy kernel will reduce to the Student's t-kernel
        self.gamma = 1
        self.spatialgamma=gamma
        self.perplexity = perplexity
        self.prob = prob
        self.likelihood_type = likelihood_type
        self.encoder = buildNetwork([input_dim] + encodeLayer, type="encode", activation=activation, prob=prob)
        self.decoder = buildNetwork([z_dim + 1] + decodeLayer, type="decode", activation=activation, prob=prob)
        self.enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self.enc_sigma = nn.Sequential(nn.Linear(encodeLayer[-1], z_dim), nn.Softplus())
        self.dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self.dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self.dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        self.device = device
        self.S = torch.from_numpy(S).to(device=device, dtype=torch.float64)  # 空间位置信息
        self.nb_loss = NBLoss().to(self.device)
        self.zinb_loss = ZINBLoss().to(self.device)
        self.pos_loss = PoissonLoss().to(self.device)
        # Initialize T as a learnable parameter
        self.T = nn.Parameter(torch.randn(encodeLayer[-1], z_dim + 1))
        N, J = X.shape
        self.attention_spatial_weight = CrossAttentionSpatialWeight(J, 2, hidden_dim=100, output_dim=N)  # 使用新的注意力机制类
        self.gdc_layers = None  # Initialized later in simulate_diffusion_process
        self.diff_to_hyp = DiffusionToHyperbolic(input_dim=encodeLayer[-1], output_dim=z_dim + 1)
        # 延迟初始化 GraphTransformationLayer 直到我们知道 in_features
        self.num_relations = 1
        self.gtn_layer = None

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def aeForward(self, x, transition_matrix):
        """
        z_mu：这是模型的均值输出，通常用于表示数据点在双曲空间的预期或平均位置。
        z：这是通过从以 z_mu 为均值的分布中采样得到的，表示具体的样本点，用于解码器或后续处理中。

        """

        x = x.float()  # 确保输入数据x为Float类型
        h = self.encoder(x)

        tmp = self.enc_mu(h)
        z_mu = self._polar_project(tmp).clone()  # 使用.clone()确保没有就地修改

        z_sigma_square = self.enc_sigma(h).clamp(min=1e-6, max=15).clone()  # 同上
        q_z = HyperbolicWrappedNorm(z_mu, z_sigma_square)
        z = q_z.sample()
        z_transformed = torch.matmul(z, self.T.T)  # 更新z_mu
        # Apply transformation matrix T
        z_updated = torch.matmul(transition_matrix, z)
        # 计算基于特征的邻接矩阵
        # adjacency_matrix = self.compute_adjacency_matrix_from_features(h)

        # 应用扩散映射到双曲映射的整合
        # diffusion_output = self.simulate_diffusion_process_output(h, adjacency_matrix)
        # z = self.diff_to_hyp(diffusion_output)  # 将扩散映射的输出转换为双曲空间坐标

        h = self.decoder(z_updated)
        #        h = self.decoder(z_transformed) #改进一
        mean = self.dec_mean(h)
        disp = self.dec_disp(h)
        pi = self.dec_pi(h)

        return q_z, z, z_mu, mean, disp, pi, z_transformed, z_updated

    # 双曲空间嵌入表示
    def _polar_project(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_unit = x / torch.clamp(x_norm, min=eps)
        x_norm = torch.clamp(x_norm, 0, 32)
        z = torch.cat((torch.cosh(x_norm), torch.sinh(x_norm) * x_unit), dim=1)
        return z

    def tsne_repel(self, z, s, p):
        n = z.size()[0]
        ### pairwise distances
        #        num = lorentz_distance_mat(z, z)**2
        #        num = torch.pow(1.0 + num, -1)
        # num = (lorentz_distance_mat(z, z)/self.gamma)**2
        num = (self.change_lorentz_distance_mat(z, z, s) / self.gamma) ** 2

        num = 1 / self.gamma / (1.0 + num)
        p = p / torch.unsqueeze(torch.sum(p, dim=1), 1)

        attraction = p * torch.log(num)
        attraction = -torch.sum(attraction)

        den = torch.sum(num, dim=1) - 1
        repellant = torch.sum(torch.log(den))

        return (repellant + attraction) / n

    def KLD(self, q_z, z):
        loc = torch.cat((torch.ones(z.shape[0], 1), torch.zeros(z.shape[0], self.z_dim)), dim=-1).to(self.device)
        p_z = HyperbolicWrappedNorm(loc, torch.ones(z.shape[0], self.z_dim).to(self.device))

        kl = q_z.log_prob(z) - p_z.log_prob(z)
        return torch.mean(kl)


    def encodeBatch(self, X):
        """
        Output latent representations and project to 2D Poincare ball for visualization
        """

        self.to(self.device)

        encoded = []
        self.eval()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / self.batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx * self.batch_size: min((batch_idx + 1) * self.batch_size, num)]
            sbatch = self.S[batch_idx * self.batch_size: min((batch_idx + 1) * self.batch_size, num)]
            inputs = Variable(xbatch)
            # 计算邻接矩阵和模拟扩散过程
            adjacency_matrix = self.compute_adjacency_matrix(xbatch, sbatch)
            transition_matrix = self.simulate_diffusion_process(adjacency_matrix)

            _, _, z, _, _, _, _, _ = self.aeForward(inputs, transition_matrix)
            z = lorentz2poincare(z)
            encoded.append(z.data.cpu().detach())

        encoded = torch.cat(encoded, dim=0)
        self.train()
        return encoded.numpy()

    def decodeBatch(self, X):
        """
        Output denoised counts
        """

        self.to(self.device)

        decoded = []
        self.eval()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / self.batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx * self.batch_size: min((batch_idx + 1) * self.batch_size, num)]
            sbatch = self.S[batch_idx * self.batch_size: min((batch_idx + 1) * self.batch_size, num)]
            inputs = Variable(xbatch)
            # 计算邻接矩阵和模拟扩散过程
            adjacency_matrix = self.compute_adjacency_matrix(xbatch, sbatch)
            transition_matrix = self.simulate_diffusion_process(adjacency_matrix)

            _, _, _, mean, _, _, _, _ = self.aeForward(inputs, transition_matrix)
            decoded.append(mean.data.cpu().detach())

        decoded = torch.cat(decoded, dim=0)
        self.train()
        return decoded.numpy()

    # 假设 lorentz_distance_mat 已经定义，这里我们对它进行必要的修改以整合空间信息
    def change_lorentz_distance_mat(self, x, y, s, lambda_param=0.5, alpha_param=1.0):
        """
        计算双曲空间中的距离，并整合空间位置信息进行调整。

        参数:
        - z: 双曲空间中的嵌入表示，尺寸为 [batch_size, z_dim+1]
        - s: 空间位置矩阵，尺寸为 [batch_size, 2]
        - lambda_param: 距离调整强度的超参数
        - alpha_param: 控制空间距离影响的衰减因子
        """
        #print(x.shape)
        # 原始的双曲洛伦兹距离计算
        dist_z = lorentz_distance_mat(x, y)  # original_lorentz_distance_mat 为已存在的双曲距离计算函数
        # 计算物理空间中的距离
        dist_s = torch.cdist(s, s, p=2)
        # 距离调整公式
        adjusted_dist_z = dist_z * (1 + lambda_param * torch.exp(-alpha_param * dist_s))
        return adjusted_dist_z

    # 在scDHMap类中增加空间正则化损失计算
    def spatial_reg_loss(self, z_mu, s, w_ij, z_transformed, alpha=0.1):
        """
        计算空间位置信息正则化损失
        Args:
            z_mu: 双曲空间中的嵌入表示，尺寸为 [batch_size, z_dim]
            s: 空间位置矩阵，尺寸为 [batch_size, 2]
            w_ij: 空间相似性权重矩阵
            z_transformed:N*J的矩阵
        Returns:
            loss: 空间位置信息正则化损失
        """
        # 使用双曲距离计算嵌入表示之间的距离
        # dist_z = lorentz_distance_mat(z_mu, z_mu)
        dist_z = self.change_lorentz_distance_mat(z_mu, z_mu, s)
        # 计算物理空间中的距离，并应用对数映射函数进行调整
        dist_s = torch.cdist(s, s, p=2)
        mapped_dist_s = torch.log(1 + alpha * dist_s)  # 应用对数映射函数进行距离调整
        # 使用相似性权重调整映射后距离的差异
        #    loss = torch.sum(w_ij *(dist_z - mapped_dist_s) ** 2)* average_correlation_weight
        loss = torch.sum(w_ij * (dist_z - mapped_dist_s) ** 2)
        return loss



    def compute_adjacency_matrix(self, x, s, alpha=1.0):
        """
        根据基因表达数据x和空间位置信息s计算邻接矩阵，完全在PyTorch中实现。
        这里的相似性计算综合了基因表达数据和空间位置信息。
        """
        # x_similarity = self.compute_cosine_similarity(x)  # 基因表达数据的余弦相似性
        x_dist = torch.cdist(x, x, p=2)  # 基因表达数据的距离
        s_dist = torch.cdist(s, s, p=2)  # 空间位置的距离

        # 使用高斯核函数计算基于距离的相似性
        x_similarity = torch.exp(-x_dist ** 2 / (2. * alpha ** 2)).to(self.device)
        s_similarity = torch.exp(-s_dist ** 2 / (2. * alpha ** 2))
        # Move s_similarity to CUDA device
        s_similarity = s_similarity.to('cuda')
        # 结合基因表达数据和空间位置信息的相似性
        adjacency_matrix = x_similarity * s_similarity  # 元素乘法
        return adjacency_matrix

    def simulate_diffusion_process(self, adjacency_matrix, tau=1.0, device='cuda'):
        """
        使用图扩散卷积模拟非线性扩散过程。
        参数:
        - adjacency_matrix: 邻接矩阵，形状为 (N, N)
        - tau: 控制扩散速率的参数
        """
        # 确保 GTN 层在第一次使用时初始化
        if self.gtn_layer is None:
            num_features = adjacency_matrix[0].size(0)
            self.gtn_layer = GraphTransformationLayer(num_features, self.num_relations)

        num_features = adjacency_matrix.size(0)

        # 归一化邻接矩阵以获得转移概率矩阵
        row_sum = adjacency_matrix.sum(dim=1, keepdim=True)
        norm_adj = (adjacency_matrix / row_sum).to(torch.float32)

        # 初始化图卷积层
        self.gdc_layers = nn.ModuleList(
            [GraphDiffusionConvolution(num_features, num_features) for _ in range(3)])
        # 初始化特征为单位矩阵，模拟每个节点的独立特征
        x = torch.eye(num_features).to(adjacency_matrix.device)
        # 逐层应用图扩散卷积
        for layer in self.gdc_layers:
            x = F.relu(layer(x, norm_adj))
           #x = layer(x, norm_adj) + x
        # 应用tau参数调节扩散效果
        diffusion_matrix = torch.exp(-tau * (1.0 - x))
        return diffusion_matrix

    def calculate_diffusion_loss(self, z, transition_matrix, z_mu, s):
        """
        根据转移矩阵计算扩散过程的损失。
        这里使用转移矩阵与实际的双曲空间距离之间的差异作为损失，
        以优化双曲空间嵌入表示。

        # 首先，基于双曲空间计算点对之间的距离
        dist_z = lorentz_distance_mat(z, z)  # 双曲距离计算

        # 根据扩散过程的结果（转移矩阵）调整损失计算
        # 注意：这里将转移矩阵视为反映细胞间信息传播强度的矩阵
        # 损失计算将基于转移矩阵与双曲距离之间的差异
        transition_matrix = transition_matrix.to(z.device)  # 确保转移矩阵在正确的设备上
        loss = torch.mean((dist_z - transition_matrix) ** 2)
        return loss
    """
        """
           使用改进后的KL散度计算扩散损失。
           """
        epsilon = 1e-8  # 一个小的常数，防止概率为零
        dist_z = self.change_lorentz_distance_mat(z_mu, z_mu, s)
        dist_z = dist_z - torch.max(dist_z, dim=1, keepdim=True)[0]
        dist_z = torch.clamp(dist_z, min=0)  # 确保距离不为负
        transition_matrix = transition_matrix - torch.max(transition_matrix, dim=1, keepdim=True)[0]
        q_ij = F.softmax(-dist_z, dim=1)
        p_ij_normalized = F.softmax(transition_matrix, dim=1)
        # 确保概率和为1
        q_ij = q_ij / q_ij.sum(dim=1, keepdim=True)
        # 将所有张量移动到z所在的设备上，确保计算可以在同一设备上执行
        p_ij_normalized = torch.clamp(p_ij_normalized.to(self.device), min=epsilon, max=1.0)
        q_ij = q_ij.to(self.device)
        # 计算KL散度
        kl_loss = F.kl_div(q_ij.log(), p_ij_normalized, reduction='batchmean')

        #kl_loss = F.kl_div(p_ij_normalized.log(), q_ij, reduction='batchmean')
        return kl_loss

    def pretrain_autoencoder(self, X, X_raw, size_factor, S, lr=0.001, pretrain_iter=200):
        """
        Pretrain the model with ZINB/NB hyperbolic VAE only and spatial regularization.

        Parameters:
        -----------
        X: array_like, shape (n_samples, n_features)
            The normalized raw counts
        X_raw: array_like, shape (n_samples, n_features)
            The raw counts, which need for the ZINB/NB loss
        size_factor: array_like, shape (n_samples)
            The size factor of each sample, which need for the ZINB/NB loss
        S: array_like, shape (n_samples, 2)
            The spatial coordinates of each sample
        lr: float, defalut = 0.001
            Learning rate for the optimizer
        pretrain_iter: int, default = 400
            Pretrain iterations
        ae_save: bool, default = True
            Whether to save the pretrained weights
        ae_weights: str
            Directory name to save the model weights
        sigma_spatial: float
            The sigma parameter for the spatial weight calculation, default is 1.0
        """
        ##进行对应的注意力机制处理

        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay,
        #                       amsgrad=True)

        print("Pretraining stage")
        optimizer = optim.Adam(
            list(self.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=True
        )
        for epoch in range(pretrain_iter):
            self.zero_grad()
            loss_reconn_val = 0
            loss_kld_val = 0
            loss_spatial_val = 0  # For accumulating spatial regularization loss
            loss_val = 0

            ###
            self.to(self.device)
            num = X.shape[0]

            dataset = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(size_factor), torch.Tensor(S))
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

            for batch_idx, (x_batch, x_raw_batch, sf_batch, s_batch) in enumerate(dataloader):
                self.zero_grad()  # 清除现有梯度
                x_tensor = x_batch.to(self.device)
                x_raw_tensor = x_raw_batch.to(self.device)
                sf_tensor = sf_batch.to(self.device)
                s_tensor = s_batch.to(self.device)  # Convert spatial coordinates to tensor

                # 使用注意力机制计算空间权重
                w_ij, Rex = self.attention_spatial_weight(x_tensor, s_tensor)
                # 计算邻接矩阵和模拟扩散过程
                adjacency_matrix = self.compute_adjacency_matrix(x_batch, s_batch)
                transition_matrix = self.simulate_diffusion_process(adjacency_matrix)

                q_z, z, z_mu, mean_tensor, disp_tensor, pi_tensor, z_transformed, z_updated = self.aeForward(x_tensor,
                                                                                                             transition_matrix)
                if self.likelihood_type == "nb":
                    loss_reconn = self.nb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor,
                                               scale_factor=sf_tensor)
                elif self.likelihood_type == "zinb":
                    loss_reconn = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor,
                                                 scale_factor=sf_tensor)
                elif self.likelihood_type == "Possion":
                    loss_reconn = self.pos_loss(x=x_raw_tensor, mean=mean_tensor, scale_factor=sf_tensor)
                else:
                    raise Exception("The likelihood type must be one of 'zinb' or 'nb'")

                # 计算扩散过程的损失并加入到总损失中
                diffusion_loss = self.calculate_diffusion_loss(z, transition_matrix, z_updated, s_tensor)
                loss_kld = self.KLD(q_z, z)
                loss_spatial = self.spatial_reg_loss(z_mu, s_tensor, w_ij,
                                                     z_transformed)  # Calculate spatial regularization loss

                loss = loss_reconn + loss_kld + loss_spatial + diffusion_loss
                loss.backward()
                optimizer.step()

                loss_reconn_val += loss_reconn.item() * len(x_batch)
                loss_kld_val += loss_kld.item() * len(x_batch)
                loss_spatial_val += loss_spatial.item() * len(x_batch)
                loss_val += loss.item() * len(x_batch)

            loss_reconn_val = loss_reconn_val / num
            loss_kld_val = loss_kld_val / num
            loss_spatial_val = loss_spatial_val / num
            loss_val = loss_val / num
            print(
                f'Pretraining epoch {epoch + 1}, Total loss: {loss_val:.8f}, Reconn loss: {loss_reconn_val:.8f}, KLD loss: {loss_kld_val:.8f}, Spatial loss: {loss_spatial_val:.8f}')

    def train_model(self, X, X_raw, size_factor, X_pca, S,  lr=0.001, maxiter=1000, minimum_iter=0,patience=100):

        """
        Train the model with the ZINB/NB hyperbolic VAE and the hyberbolic t-SNE regularization.

        Parameters:
        -----------
        X: array_like, shape (n_samples, n_features)
            The normalized raw counts
        X_raw: array_like, shape (n_samples, n_features)
            The raw counts, which need for the ZINB/NB loss
        size_factor: array_like, shape (n_samples)
            The size factor of each sample, which need for the ZINB/NB loss
        X_pca: array_like, shape (n_samples, n_PCs)
            The principal components of the analytic Pearson residual normalized raw counts
        X_true_pca: array_like, shape (n_samples, n_PCs)
            The principal components of the analytic Pearson residual normalized true counts
            This is used for evaluation of simulation experiments; for real data, it can be set to None
        lr: float, defalut = 0.001
            Learning rate for the opitimizer
        maxiter: int, default = 5000
            Maximum number of iterations
        minimum_iter: int, default = 0
            Minimum number of iterations
        patience: int, default = 150
            Patience for the early stop
        save_dir: str
            Directory name to save the model weights
        """

        self.to(self.device)
        X = torch.tensor(X, dtype=torch.float)
        X_raw = torch.tensor(X_raw, dtype=torch.float)
        size_factor = torch.tensor(size_factor, dtype=torch.float)
        num = X.shape[0]
        sample_indices = np.arange(num)
        num_batch = int(math.ceil(1.0 * num / self.batch_size))

        perplexity = np.array(self.perplexity).astype(np.double)
        rng = np.random.default_rng(42)  # 使用固定种子的随机数生成器


        print("Training...")

        early_stopping = EarlyStopping(patience=patience)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay,
                               amsgrad=True)
        train_losses = []
        val_losses = []
        kl_divergences = []
        recon_losses = []
        spatial_losses=[]
        diffusion_losses=[]
        for epoch in range(maxiter):

            ###
            loss_reconn_val = 0
            loss_tsne_val = 0
            loss_kld_val = 0
            loss_val = 0
            loss_spatial = 0
            diffusion_loss = 0
            rng.shuffle(sample_indices)
            for batch_idx in range(num_batch):
                batch_indices = sample_indices[batch_idx * self.batch_size: min((batch_idx + 1) * self.batch_size, num)]

                x_batch = X[batch_indices]
                x_raw_batch = X_raw[batch_indices]
                sf_batch = size_factor[batch_indices]
              #  x_tensor = torch.tensor(x_batch, dtype=torch.float).to(self.device)
              #  x_raw_tensor = torch.tensor(x_raw_batch, dtype=torch.float).to(self.device)
              #  sf_tensor = torch.tensor(sf_batch, dtype=torch.float).to(self.device)
                x_tensor = x_batch.clone().detach().to(self.device)
                x_raw_tensor = x_raw_batch.clone().detach().to(self.device)
                sf_tensor = sf_batch.clone().detach().to(self.device)

                dist_X_pca_batch = pairwise_distances(X_pca[batch_indices], metric="euclidean").astype(np.double)
                p_batch = compute_gaussian_perplexity(dist_X_pca_batch, perplexities=perplexity)
                p_batch = torch.tensor(p_batch)
                #p_tensor = torch.tensor(p_batch, dtype=torch.float).to(self.device)
 #               p_tensor = torch.tensor(p_batch, dtype=torch.float).clone().detach().to(self.device)
                p_tensor = p_batch.clone().detach().to(self.device)

                s_batch = S[batch_indices]
#                s_tensor = torch.tensor(s_batch, dtype=torch.float).to(self.device)
                s_tensor = torch.tensor(s_batch, dtype=torch.float).clone().detach().to(self.device)

                w_ij, Rex = self.attention_spatial_weight(x_tensor, s_tensor)  # 基于注意力机制得到权重wij和结合空间位置信息ReX
                # 计算邻接矩阵和模拟扩散过程
                adjacency_matrix = self.compute_adjacency_matrix(x_tensor, s_tensor)
                transition_matrix = self.simulate_diffusion_process(adjacency_matrix)

                # q_z, z, z_mu, mean_tensor, disp_tensor, pi_tensor,z_transformed,z_updated = self.aeForward(x_tensor,transition_matrix)
                q_z, z, z_mu, mean_tensor, disp_tensor, pi_tensor, z_transformed, z_updated = self.aeForward(x_tensor,
                                                                                                             transition_matrix)
                if self.likelihood_type == "nb":
                    loss_reconn = self.nb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor,
                                               scale_factor=sf_tensor)
                else:
                    raise Exception("The likelihood type must be one of 'zinb' or 'nb'")

                # 计算扩散过程的损失并加入到总损失中
                diffusion_loss = self.calculate_diffusion_loss(z, transition_matrix, z_updated, s_tensor)
                loss_tsne = self.tsne_repel(z_mu, s_tensor, p_tensor)
                loss_kld = self.KLD(q_z, z)

                loss = loss_reconn + self.alpha * loss_tsne + self.beta * loss_kld + diffusion_loss
                #loss = loss_reconn + self.alpha * loss_tsne + diffusion_loss
                # 计算空间位置信息正则化损失
                loss_spatial = self.spatial_reg_loss(z_mu, s_tensor, w_ij, z_transformed)

                # 更新总损失函数，加入空间位置信息正则化损失
                loss += self.spatialgamma * loss_spatial  # 假设空间正则化的权重为gamma

                self.zero_grad()
                loss.backward()
                optimizer.step()

                loss_reconn_val += loss_reconn.item() * len(x_batch)
                loss_tsne_val += loss_tsne.item() * len(x_batch)
                loss_kld_val += loss_kld.item() * len(x_batch)
                loss_val += loss.item() * len(x_batch)
                loss_spatial += loss_spatial.item() * len(x_batch)
                diffusion_loss += diffusion_loss.item() * len(x_batch)
            loss_reconn_val = loss_reconn_val / num
            loss_tsne_val = loss_tsne_val / num
            loss_kld_val = loss_kld_val / num
            loss_val = loss_val / num
            loss_spatial = loss_spatial / num
            diffusion_loss = diffusion_loss / num

            train_losses.append(loss_val)
            val_losses.append(loss_reconn_val)
            kl_divergences.append(loss_kld_val)
            recon_losses.append(loss_tsne_val)
            spatial_losses.append(loss_spatial)
            diffusion_losses.append(diffusion_losses)

            print(
                'Training epoch {}, Total loss:{:.8f}, reconn loss:{:.8f}, t-SNE loss:{:.8f}, KLD loss:{:.8f}，Spatial loss:{:.8f},diffusion loss:{:.8f}'
                .format(epoch + 1, loss_val, loss_reconn_val, loss_tsne_val, loss_kld_val, loss_spatial,
                        diffusion_loss))

            if epoch + 1 >= minimum_iter:
                early_stopping(loss_tsne_val, self)
                if early_stopping.early_stop:
                    print('EarlyStopping: run {} iteration'.format(epoch))
                    break
        return train_losses, val_losses, kl_divergences, recon_losses,spatial_losses,diffusion_losses