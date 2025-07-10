# HyperDiffuseNet

HyperDiffuseNet是一种结合双曲几何、图扩散卷积网络和深度变分自编码技术的机器学习框架，专为高维数据的降维和表示学习设计。该框架能有效捕获数据的层次结构和空间关系，适用于各类复杂数据分析任务。

## 功能特点

- **双曲空间嵌入**：利用双曲几何的负曲率特性，在低维表示中更好地保留数据的层次结构
- **图扩散卷积**：通过图结构建模数据间的相互依赖，增强空间一致性
- **空间注意机制**：动态整合特征信息与空间信息，优化相似性计算
- **多重正则化**：结合重建误差、空间一致性损失、t-SNE损失和KL散度，优化表示质量
- **统一学习框架**：同时考虑特征相似度和空间近邻关系，实现高质量的低维表示

## 安装指南

### 系统要求

- Python 3.7或更高版本
- CUDA支持的NVIDIA GPU（推荐但非必需）
- 16GB以上内存（建议）

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/Jinsl-lab/HyperDiffuseNet.git
   cd HyperDiffuseNet
  
2.创建并激活虚拟环境（建议）：
    # 使用conda
conda create -n hyperdiffuse python=3.8
conda activate hyperdiffuse

  
3.安装依赖项：
pip install -r requirements.txt

4.验证安装：
python test_installation.py


5.快速开始
基本使用
import numpy as np
import torch
from HyperDiffuseNet_HyperSpatial_Attention import HyperDiffuseNet
from preprocess import read_dataset, normalize, pearson_residuals

# 准备数据
# X: 特征矩阵, 形状为 [n_samples, n_features]
# spatial_coords: 空间坐标 (可选), 形状为 [n_samples, 2]

# 标准化数据
adata = read_dataset(sc.AnnData(X))
adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)

# 初始化模型
model = HyperDiffuseNet(
    X=adata.X,
    S=spatial_coords,  # 可选
    input_dim=adata.n_vars,
    encodeLayer=[128, 64, 32],
    decodeLayer=[32, 64, 128],
    z_dim=2,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 训练模型
model.pretrain_autoencoder(
    X=adata.X,
    X_raw=adata.raw.X,
    size_factor=adata.obs.size_factors,
    S=spatial_coords
)

model.train_model(
    X=adata.X,
    X_raw=adata.raw.X,
    size_factor=adata.obs.size_factors,
    X_pca=pca_result,
    S=spatial_coords
)

# 获取低维嵌入
embeddings = model.encodeBatch(torch.tensor(adata.X))
有关更详细的使用说明，请参阅examples目录中的示例脚本：

examples/basic_usage.py: 基本使用示例
examples/advanced_analysis.py: 高级分析示例

主要模块

HyperDiffuseNet_HyperSpatial_Attention.py: 核心模型实现
layers.py: 损失函数和网络层定义
lorentzian_helper.py: 双曲空间数学工具
wrapped_normal.py: 双曲分布包装器
preprocess.py: 数据预处理函数
tsne_helper.py: t-SNE相关实现
single_cell_tools.py: 实用工具函数

参数调整
关键参数及其建议值：

网络结构参数:

encodeLayer & decodeLayer: 编码器和解码器的隐藏层配置
z_dim: 嵌入空间维度，推荐2-3(可视化)或8-16(特征提取)


训练参数:

batch_size: 批量大小，小数据集32-128，大数据集512或更大
lr: 学习率，推荐1e-3到1e-4


损失函数权重:

alpha: t-SNE损失权重，范围100-1000
beta: KL散度权重，范围5-50
gamma: 空间正则化权重，范围0.1-10
