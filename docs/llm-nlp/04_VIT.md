
# Vision Transformer (ViT) 架构与原理精讲笔记
[📄 点击下载/预览笔记PDF](../../assets/04_ViT论文.pdf)
## 目录

-   [一、 图像 Patch 化与线性映射（Patch Embedding）](https://www.google.com/search?q=%23%E4%B8%80-%E5%9B%BE%E5%83%8F-patch-%E5%8C%96%E4%B8%8E%E7%BA%BF%E6%80%A7%E6%98%A0%E5%B0%84patch-embedding)
    
-   [二、 `[CLS]` Token 分类标记机制](https://www.google.com/search?q=%23%E4%BA%8C-cls-token-%E5%88%86%E7%B1%BB%E6%A0%87%E8%AE%B0%E6%9C%BA%E5%88%B6)
    
-   [三、 ViT 位置编码（Position Embedding）](https://www.google.com/search?q=%23%E4%B8%89-vit-%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81position-embedding)
    
-   [四、 多头自注意力机制（Multi-Head Attention & 维度变化）](https://www.google.com/search?q=%23%E5%9B%9B-%E5%A4%9A%E5%A4%B4%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6multi-head-attention--%E7%BB%B4%E5%BA%A6%E5%8F%98%E5%8C%96)
    
-   [五、 ViT 主干结构与网络变体](https://www.google.com/search?q=%23%E4%BA%94-vit-%E4%B8%BB%E5%B9%B2%E7%BB%93%E6%9E%84%E4%B8%8E%E7%BD%91%E7%BB%9C%E5%8F%98%E4%BD%93)
    
-   [六、 论文实验与核心结论分析](https://www.google.com/search?q=%23%E5%85%AD-%E8%AE%BA%E6%96%87%E5%AE%9E%E9%AA%8C%E4%B8%8E%E6%A0%B8%E5%BF%83%E7%BB%93%E8%AE%BA%E5%88%86%E6%9E%90)
    

## 一、 图像 Patch 化与线性映射（Patch Embedding）

ViT 的核心思想是将图像转换为类似于 NLP 中的 Token 序列输入。

### 1. 基础转换过程（以 $224 \times 224 \times 3$ 图像为例）

-   **输入图像**：$224 \times 224 \times 3$。
    
-   **Patch 切割**：设定每个 Patch 大小为 $16 \times 16 \times 3$。
    
-   **Patch 数量**：
    
    $$\left(\frac{224}{16}\right) \times \left(\frac{224}{16}\right) = 14 \times 14 = 196 \text{ 个}$$
    
-   **Flatten 拉平**：每个 $16 \times 16 \times 3$ 的 Patch 被拉平为长度为 $768$ 的一维向量。
    
-   **线性映射**：通过线性网络映射成维度为 $768$ 的向量，输入矩阵形状变为 $(196, 768)$。
    

### 2. 工程实现：卷积层等价转换

在实际代码实现中，通常直接使用 **二维卷积（Conv2d）** 一步完成切片与线性映射：

-   **卷积核大小（Kernel Size）**：$16 \times 16$
    
-   **卷积核数量（Out Channels）**：$768$
    
-   **步长（Stride）**：$16$
    
-   **填充（Padding）**：$0$
    
-   **输出特征图维度**：$14 \times 14 \times 768$，展平后即为 $(196, 768)$。
    

> **卷积实现的优点**：
> 
> 1.  **信息浓缩**：将 $16 \times 16 \times 3 = 768$ 个像素信息提取并浓缩为 $768$ 维高语义特征。
>     
> 2.  **并行高效**：GPU 可并行计算所有卷积核与位置，速度快。
>     
> 3.  **自动学习**：$768$ 个卷积核参数通过反向传播自动学习，寻找最佳特征模式。
>     

## 二、 `[CLS]` Token 分类标记机制

类似于 BERT，ViT 引入了一个可学习的分类标记（`[CLS]` Token）。

### 1. 本质与属性

-   **本质**：一个可学习参数向量 $\mathbf{c} \in \mathbb{R}^{1 \times 768}$，不来源于图像内容。
    
-   **初始化**：随机初始化，维度与 Patch Embedding 一致（$768$）。
    
-   **复用**：每个 Batch 复制同一个 `[CLS]` Token，通过反向传播更新参数。
    

### 2. 构建流程

1.  **初始化**：创建可学习向量 $\mathbf{c}$，维度 $(1, 768)$。
    
2.  **拼接序列**：与图像 Patch 序列 $(196, 768)$ 进行 Concat 操作，获得 $197 \times 768$ 维度的向量。
    
3.  **特征提取**：数据输入 Transformer Encoder 处理，输出形状保持不变，仍为 $(197, 768)$。
    
4.  **分类输出**：提取第 $0$ 个 Token（即 `[CLS]` 对应的输出向量）作为整张图像的全局特征表征，送入 MLP Head 进行分类预测。
    

## 三、 ViT 位置编码（Position Embedding）

### 1. 为什么需要位置编码？

ViT 将图像切分为 1D 序列后，自注意力机制（Self-Attention）具备置换不变性，如果不额外告诉模型“每个 Patch 在图像中的位置”，模型很难区分空间结构。

### 2. 位置编码设计

-   **形式**：引入可学习的位置编码矩阵 $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$（如 $197 \times 768$）。
    
-   **加入方式**：位置编码直接与 Patch Embedding 相加：
    
    $$\mathbf{Z}_0 = \mathbf{Tokens} + \mathbf{E}_{\text{pos}}$$
    
-   **效果对比**：原论文实验表明，1D 可学习位置编码（Standard Learnable 1D Position Embeddings）效果显著优于不加位置编码（准确率从 ~0.613 提升至 ~0.642），而更复杂的 2D 或相对位置编码收益提升并不明显。
    

## 四、 多头自注意力机制（Multi-Head Attention & 维度变化）

在多头注意力机制中，张量形状的变化流程如下（以 Batch Size $= B$, Token 数 $N = 197$, 特征通道数 $C = 768$, 头数 $h = 8$ 或 $12$ 为例）：

Plaintext

```
输入特征 X: [B, N, C] (例: [B, 197, 768])
  │
  ├──> 通过 Linear (dim -> dim * 3) 生成 QKV 矩阵 -> [B, N, 3, h, d]  (d = C / h)
  │
  ├──> Permute 重排维度 -> [3, B, h, N, d]
  │
  ├──> 拆分为 Q, K, V -> 每个张量维度均为 [B, h, N, d]
  │
  ├──> Attention 计算:
  │      1. Score = Q @ K.T   -> [B, h, N, d] @ [B, h, d, N] = [B, h, N, N]
  │      2. Attn  = Softmax(Score / sqrt(d))
  │      3. Out   = Attn @ V   -> [B, h, N, N] @ [B, h, N, d] = [B, h, N, d]
  │
  ├──> Transpose + Reshape 还原维度 -> [B, N, C]
  │
  └──> 通过 Projection Linear 层输出 -> [B, N, C]

```

### PyTorch 核心代码实现

Python

```
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 一次性线性变换生成 Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        # [B, N, 3*C] -> [B, N, 3, num_heads, head_dim] -> [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # [B, h, N, d] @ [B, h, d, N] -> [B, h, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # [B, h, N, N] @ [B, h, N, d] -> [B, h, N, d] -> [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

```

## 五、 ViT 主干结构

### 1. Encoder 块的计算流（Pre-LN 架构）

单个 Encoder Block 的输入输出流动格式如下：

1.  $X_{1} = X + \text{MultiHeadAttention}(\text{LayerNorm}(X))$
    
2.  $X_{2} = X_{1} + \text{MLP}(\text{LayerNorm}(X_{1}))$
    

其中 **MLP 层** 通常将维度从 $768$ 放大 $4$ 倍至 $3072$，再通过 GELU 激活函数压缩回 $768$。



## 六、 论文实验与核心结论分析

### 1. 预训练数据集规模的影响

-   **小数据阶段**：当数据集较小（如 ImageNet-1k）且缺少强正则化时，ViT 的表现逊于同等规模的 CNN（如 ResNet），因为 Transformer 缺少 CNN 原生的“归纳偏置”。
    
-   **大数据阶段**：当预训练数据集放大至 **ImageNet-21k** 或 **JFT-300M** 时，ViT 的大容量特征表达能力得以全面释放，性能显著超越最强 CNN 基线。
    

### 2. 核心结论摘要

1.  **预训练规模至关重要**：大规模数据预训练是 ViT 获得强泛化能力的关键。
    
2.  **Patch 粒度敏感度**：Patch 尺寸越小（如 $/14$ 比 $/16$），Token 数量越多，保留的细粒度细节越丰富，准确率更高，但计算开销显著增加。
    
3.  **算力性价比突出**：在达到相同或超越传统强 CNN（如 BiT-L, EfficientNet）的准确率时，ViT 训练所需的 Compute/TPU core-days 大幅降低，展示出了出色的算力效率。