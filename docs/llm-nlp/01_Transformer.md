
# Transformer 架构与原理精讲笔记
[📄 点击下载/预览笔记PDF](../../assets/01_transformer.pdf)
> **文档说明**：本笔记系统梳理了 Transformer 神经网络架构的核心原理、数学公式、关键模块细节以及代码实现。旨在为深度学习研究者与开发者提供一份结构清晰、条理分明的全面参考指南。

## 1. 概述与总体架构

### 1.1 背景与设计动机

在时间序列与自然语言建模中，传统循环神经网络（RNN）难以进行大规模并行计算，且在建模长距离依赖时容易出现梯度消失或爆炸。Transformer 架构的提出主要为了解决以下两个核心问题：

-   **提升训练并行性**：摆脱 RNN 依时间步顺序计算的限制，利用矩阵并行运算极大加速训练。
    
-   **长距离依赖建模**：引入自注意力机制，使序列中任意两个 Token 之间的计算路径长度缩短为 $\mathcal{O}(1)$。
    

### 1.2 三大核心机制

-   **注意力机制（Attention Mechanism）**：提取上下文语义关系，动态计算序列内任意位置之间的关联。
    
-   **位置编码（Positional Encoding）**：为无时序偏置的 Self-Attention 补充 Token 的绝对与相对位置信息。
    
-   **残差连接与层归一化（Residual & LayerNorm）**：构建极深网络的梯度高速公路，稳定训练过程。
    

### 1.3 整体架构拓扑

Transformer 整体采用典型的 **编码器-解码器（Encoder-Decoder）** 结构：

```
                  Outputs Probabilities
                            ↑
                         Softmax
                            ↑
                         Linear
                            ↑
                    ┌──────────────┐
                    │   Decoder    │×N  <─── (编码信息 Q, K 来自 Encoder)
                    └──────────────┘
                           ↑
                    ┌──────────────┐
                    │   Encoder    │×N
                    └──────────────┘
                           ↑
                 Positional Encoding
                           ↑
                  Input/Output Embedding
                           ↑
                    Inputs / Outputs

```

#### 核心组件一览

1.  **词嵌入向量（Embedding）**：将离散 Token 映射为连续高维稠密向量。
    
2.  **位置编码（Positional Encoding）**：提供时序与位置感知信息。
    
3.  **多头自注意力（Multi-Head Attention）**：无掩码全局自注意力，用于编码器。
    
4.  **带掩码的多头自注意力（Masked Multi-Head Attention）**：带因果掩码，用于解码器防止未来信息泄露。
    
5.  **多头交叉注意力（Cross Attention）**：连接 Encoder 与 Decoder，融合源序列与目标序列特征。
    
6.  **逐位置前馈神经网络（Feed-Forward Network, FFN）**：非线性特征变换。
    
7.  **残差连接与层归一化（Add & Norm）**：保持梯度顺畅传播，防止梯度消失。
    
8.  **输出线性层与 Softmax**：映射至词表维度，输出概率分布。
    

## 2. 输入表达与数据预处理

### 2.1 词嵌入 (Embedding)

自然语言处理需要将离散文本转换为数值矩阵，经典处理流程如下：

1.  **切词 (Tokenization)**：将字符串拆分为 Token（如 `[我, 是, 一条, 狗]`）。
    
2.  **构建词表 (Vocabulary)**：统计词频，建立大小为 $V$ 的词表索引。
    
3.  **独热编码 (One-Hot)**：
    
    -   我 $\rightarrow [1, 0, 0, 0, 0, \dots]$
        
    -   是 $\rightarrow [0, 1, 0, 0, 0, \dots]$
        
    
    > **独热编码缺点**：维度过高（$V \approx 50\text{k}+$），极度稀疏，且无法反映语义相似度。
    
4.  **嵌入映射 (Embedding Matrix)**：
    
    通过矩阵乘法 $X_{\text{dense}} = X_{\text{one-hot}} \times W_{\text{embed}}$，将高维稀疏向量映射为低维稠密向量 $d$（论文中 $d_{\text{model}} = 512$）。
    
    -   嵌入矩阵是**可学习的参数表**。
        
    -   能有效捕获语义近远关系（如“棒球/足球”聚类、“我/你/他”聚类）。
        

### 2.2 位置编码 (Positional Encoding)

由于自注意力机制是置换不变的（无感知顺序能力），必须显式注入位置信息。Transformer 使用正弦与余弦函数的正交位置编码：

#### 计算公式

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d}}}\right)$$

-   $pos$：Token 在序列中的位置索引（$pos = 0, 1, 2, \dots$）。
    
-   $d$：模型维度（$d_{\text{model}}$）。
    
-   $i$：通道索引（$i = 0, 1, \dots, \frac{d}{2}-1$）。
    

最终模型的输入特征为两者直接叠加：$\text{Input} = X_{\text{embed}} + PE$。

## 3. 注意力机制核心原理

### 3.1 自注意力机制 (Self-Attention)

#### 核心目的

通过计算序列中每个 Token 与其他所有 Token 的相关程度，融合全句上下文更新当前特征。

#### 计算步骤（向量视角）

以输入 $a^1, a^2, a^3, a^4$ 为例：

1.  **投影生成 $q, k, v$**：
    
    -   $q^1 = W^q a^1$
        
    -   $k^i = W^k a^i, \quad v^i = W^v a^i \quad (i \in \{1, 2, 3, 4\})$
        
2.  **计算相似度得分**：$\alpha_{1,i} = q^1 \cdot k^i$
    
3.  **归一化注意力权重**：$\alpha'_{1,i} = \text{Softmax}(\alpha_{1,i})$
    
4.  **加权加和上下文**：$b^1 = \sum_{i} \alpha'_{1,i} v^i$
    

#### 矩阵运算视角

利用矩阵运算可以最大化 GPU 的并行能力：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

-   **输入矩阵 $X$**：维度 $L \times d$ （$L$ 为序列长度）
    
-   **$Q, K, V$ 矩阵**：由 $X$ 分别乘以 $W^q, W^k, W^v$ 得出，维度均为 $L \times d_k$
    
-   **注意力权重矩阵 $A$**：$A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$，维度为 $L \times L$
    

#### ❓ 缩放因子 $\sqrt{d_k}$ 的作用

1.  **防止梯度消失**：当 $d_k$ 较大时，$QK^T$ 的点积数值会很大，导致 Softmax 函数落入梯度极小的饱和区。
    
2.  **稳定数值方差**：假设 $q$ 和 $k$ 的各元素是均值为 0、方差为 1 的独立随机变量，点积的方差为 $d_k$。除以 $\sqrt{d_k}$ 可以将方差拉回 1。
    

### 3.2 多头注意力机制 (Multi-Head Attention)

#### 定义与公式

多头注意力机制将高维特征空间切分为 $h$ 个互不重叠的子空间，并行运行注意力计算：

$$head_i = \text{Attention}(X W_i^q, X W_i^k, X W_i^v)$$

$$\text{MHA}(X) = \text{Concat}(head_1, head_2, \dots, head_h) W^O$$

-   $W^O$ 用于将多头拼接后的向量重新投影回标准模型维度 $d_{\text{model}}$。
    

#### 💡 设计直觉与思考

-   **不同头的分工**：模型在训练过程中通过梯度下降自发产生特征解耦。不同头会专注于不同的表征模式（例如短距离语法关系、长距离语义依存、实体指代等）。
    
-   **容错与冗余**：多头设计增加了模型的表达容量和“中奖概率”，即使部分 Head 出现模式重合或未学到有效信息，整体结构仍具有强鲁棒性。
    

#### 代码实现 (PyTorch)

Python

```
import torch
import torch.nn as nn
import math

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        
        # 1) 线性变换并划分为 h 个头: (batch, len, d_model) -> (batch, h, len, d_k)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        
        # 2) 缩放点积注意力
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value)
        
        # 3) 拼接多头输出并还原维度
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        
        # 4) 输出线性投影
        return self.linears[-1](x)

```

### 3.3 因果与填充掩码机制 (Masking)

#### 1. 填充掩码 (Padding Mask)

为了对齐同一批次（Batch）中不同长度的输入序列，短序列需要填充 `<PAD>` 字符。

-   **操作**：在 Softmax 前，将 Padding 位置的 Score 加上极大负数（$-10^9$），经 Softmax 计算后其注意力权重衰减为 0。
    

#### 2. 因果掩码 (Causal / Lower-Triangular Mask)

用于解码器（Decoder），防止生成当前位置时“偷看”未出现的未来 Token。

-   **矩阵掩码形式**（下三角为 1，其余为 0）：
    

$$\begin{bmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 \end{bmatrix}$$

### 3.4 交叉注意力机制 (Cross-Attention)

用于连接 Encoder 与 Decoder，实现源语言对目标语言的信息注入：

-   **Query ($Q$)**：来自 **Decoder** 前一层的输出（“我当前需要找什么信息”）。
    
-   **Key ($K$) & Value ($V$)**：来自 **Encoder** 最终的编码特征（“源文本提供了哪些语义”）。
    

$$\text{CrossAttention} = \text{Softmax}\left(\frac{Q_{\text{Decoder}} K_{\text{Encoder}}^T}{\sqrt{d_k}}\right) V_{\text{Encoder}}$$

## 4. 网络深度与关键子层

### 4.1 层归一化 (Layer Normalization) 与残差连接

#### LN 与 BN 的差异

-   **BatchNorm**：沿着 Batch 维度归一化，容易受到 Batch Size 大小的干扰，不适用于变长序列。
    
-   **LayerNorm**：沿着单样本的 Feature 维度归一化，独立于 Batch Size，是 NLP 领域的标准配置。
    

#### LayerNorm 计算公式

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

#### 代码实现：Pre-LN 残差连接子层

现代 Transformer 多倾向采用 Pre-LN 结构（先 Normalization 再过子层），相比 Post-LN 具备更稳定的训练梯度。

Python

```
class SublayerConnection(nn.Module):
    """
    带 LayerNorm 和残差连接的子层模块 (Pre-LN 架构)
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 规范化 -> 子层运算 -> Dropout -> 残差主干相加
        return x + self.dropout(sublayer(self.norm(x)))

```

### 4.2 逐位置前馈神经网络 (Feed-Forward Network)

包含两次线性变换与一次非线性激活函数，针对每个 Token 独立并行投影（可视为 $1 \times 1$ 卷积）：

$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

通常包含升维再降维的过程（例如 $512 \rightarrow 2048 \rightarrow 512$），用于增强模型的非线性表达空间。

## 5. 训练与推理全流程
### 5.1 训练与推理的核心差异

#### 1. 计算模式

-   **训练阶段**：采用全并行计算（Parallel）。
    
-   **推理阶段**：采用自回归逐词生成（Autoregressive）。
    

#### 2. 目标序列输入方式

-   **训练阶段**：将完整目标文本（Target）叠加起始符后一次性送入 Decoder。
    
-   **推理阶段**：从起始符开始，将上一步生成的 Token 逐字追加到后续输入中。
    

#### 3. 掩码机制依赖

-   **训练阶段**：强烈依赖因果掩码（Causal Mask）来强行遮挡未来位置信息。
    
-   **推理阶段**：无需主动掩盖未来，序列随时间步自然增长，主要依赖 KV Cache 降低重复计算。
    

#### 4. 计算效率与瓶颈

-   **训练阶段**：效率高，可充分发挥 GPU 的大规模矩阵并行运算优势。
    
-   **推理阶段**：受限于串行依赖，主要瓶颈在于 Memory Bandwidth（内存带宽限制）。
    

### 5.2 推理生成流程 (Autoregressive)

1.  **Encoder** 执行单次 forward 计算，将输入序列映射为语义编码特征（$K_{\text{enc}}, V_{\text{enc}}$）。
    
2.  **Decoder** 输入起始标记 `<bos>`。
    
3.  生成当前位置的概率分布，采样输出预测 Token（如 `I`）。
    
4.  将新 Token 追加到历史输入序列中（`<bos> I`），重复迭代，直至生成终止标记 `<eos>`。
    

```
[源文本: 我是一条狗] ───> Encoder ───> 编码特征 (K_enc, V_enc)
                                              │
<bos>               ───────> Decoder ────────┼───> I
<bos> I             ───────> Decoder ────────┼───> am
<bos> I am          ───────> Decoder ────────┼───> a
<bos> I am a        ───────> Decoder ────────┼───> dog
<bos> I am a dog    ───────> Decoder ────────┴───> <eos> (完成)

```

### 5.3 训练流程 (Training with Teacher Forcing)

训练时直接将完整 Ground-Truth 序列追加 `<bos>` 作为 Decoder 输入，配合 **Causal Mask**，实现所有时间步 Loss 的并行计算与反向传播。

## 6. 总结与现代演进

### 6.1 Transformer 核心要素汇总

-   **位置编码 (Positional Encoding)**
    
    -   **作用**：补充时序与位置信息，消除自注意力的置换不变性。
        
    -   **机制**：基于正弦/余弦函数进行多频率正交投影。
        
-   **自注意力机制 (Self-Attention)**
    
    -   **作用**：捕捉全局序列上下文联系，降低任意两 Token 的计算距离。
        
    -   **机制**：通过矩阵点积运算 $\text{Softmax}(QK^T / \sqrt{d_k})V$。
        
-   **多头机制 (Multi-Head)**
    
    -   **作用**：多维度、多子空间联合提取特征，提高模型表达容错能力。
        
    -   **机制**：维度切分 $d_{\text{model}} / h$ 独立并行计算，拼接后投影还原。
        
-   **层归一化与残差连接 (LayerNorm + Residual)**
    
    -   **作用**：保障极深网络的梯度稳定，避免梯度消失或爆炸。
        
    -   **机制**：采用 Pre-LN 结构与 $x + f(x)$ 残差通道设计。
        
-   **因果掩码 (Causal Mask)**
    
    -   **作用**：实现训练并行化，同时严格防止模型偷看未来的答案。
        
    -   **机制**：Softmax 前将注意力得分矩阵的上三角区域设为 $-\infty$。
        

### 6.2 现代架构的演进方向

自 Transformer 提出以来，后续大语言模型（LLMs，如 LLaMA、GPT 系列、DeepSeek 等）在其基础之上衍生出了多项关键改进：

-   **注意力机制优化**
    
    -   由 Standard MHA（多头注意力）演变为 **MQA (Multi-Query Attention)**、**GQA (Grouped-Query Attention)** 以及 **MLA (Multi-head Latent Attention)**，旨在大幅降低大模型在推理阶段对 KV Cache 的显存占用。
        
-   **位置编码演进**
    
    -   由传统绝对正弦位置编码转向 **RoPE (Rotary Position Embedding, 旋转位置编码)** 或 **ALiBi**，从而获得更优越的长文本外推（Context Extrapolation）能力。
        
-   **激活函数与归一化方式**
    
    -   标准 LayerNorm + ReLU 演化为 **RMSNorm**（降低均方根计算开销）与 **SwiGLU** 激活函数的组合，显著提升了模型的表示能力与数值计算稳定性。