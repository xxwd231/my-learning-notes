
# LLaMA 系列架构演进精讲笔记 (LLaMA 1 至 LLaMA 4 MoE)
[📄 点击下载/预览笔记PDF](../../assets/14_Llama.pdf)
## 目录

-   [一、 LLaMA 1 基础架构改进](https://www.google.com/search?q=%23%E4%B8%80-llama-1-%E5%9F%BA%E7%A1%80%E6%9E%B6%E6%9E%84%E6%94%B9%E8%BF%9B)
    
-   [二、 LLaMA 2 关键技术升级](https://www.google.com/search?q=%23%E4%BA%8C-llama-2-%E5%85%B3%E9%94%AE%E6%8A%80%E6%9C%AF%E5%8D%87%E7%BA%A7)
    
-   [三、 LLaMA 3 训练数据与架构微调](https://www.google.com/search?q=%23%E4%B8%89-llama-3-%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE%E4%B8%8E%E6%9E%B6%E6%9E%84%E5%BE%AE%E8%B0%83)
    
-   [四、 LLaMA 4 MoE 架构核心技术与前沿突破](https://www.google.com/search?q=%23%E5%9B%9B-llama-4-moe-%E6%9E%B6%E6%9E%84%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E4%B8%8E%E5%89%8D%E6%B2%BF%E7%AA%81%E7%A0%B4)
    

## 一、 LLaMA 1 基础架构改进

LLaMA 1 基于标准的 Transformer Decoder 架构，并在四个核心维度上做出了经典优化：

```
                    ┌─► 1. Pre-Normalization (使用 RMS Norm 替代 LayerNorm)
                    ├─► 2. RMS Norm (简化均值计算，大幅省算力)
[ LLaMA 1 核心修改 ] ┼─► 3. RoPE 旋转位置编码 (捕获相对位置与距离)
                    └─► 4. SwiGLU / SiLU 激活函数 (平滑曲线，梯度更稳定)

```

### 1. Pre-Normalization (预归一化)

-   **机制**：将 Normalization 从每个子层的**输出位置 (Post-Norm)** 移动到**输入位置 (Pre-Norm)**。
    
-   **动机**：当模型层数极深（数十至上百层）时，Post-Norm 极易导致梯度消失或梯度爆炸；Pre-Norm 能确保传入每一层的特征都是平稳规范的，显著提升深层大模型训练的稳定性。
    

### 2. RMS Norm (Root Mean Square Normalization)

-   **机制**：传统 LayerNorm 需要计算均值（Mean）和方差（Variance）并进行“中心化+缩放”。RMS Norm 发现“中心化”步骤对大模型并非必需，因此直接省略均值计算，仅保留均方根缩放。
    
-   **数学公式**：
    
    $$\bar{a}_i = \frac{a_i}{\text{RMS}(\mathbf{a})} g_i, \quad \text{其中 } \text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} a_i^2 + \epsilon}$$
    
-   **动机**：精简计算步骤，节省时间和算力，同时模型的表达精度完全不下降。
    

### 3. RoPE (Rotary Position Embedding) 旋转位置编码

-   **机制**：将二维向量在复平面上旋转角度 $m\theta$，通过旋转矩阵 $R(m\theta)$ 对 Query ($q$) 和 Key ($k$) 向量施加位置变换。
    
    ```
    q·k (无位置信息)  ───►  q R(m) · k R(n) = q R(m - n) kᵀ (带有相对位置信息)
    
    ```
    
-   **本质推导**：
    
    $$q R(m) \cdot k R(n) = q R(m) R(n)^T k^T = q R(m - n) k^T$$
    
    通过点乘后的结果可知，注意力权重仅与两个 Token 之间的**相对距离 $(m-n)$** 挂钩。
    
-   **动机**：完美兼顾绝对位置与相对位置关系，并且具备优秀的长度外推能力，在处理超长文本时表现远优于绝对位置编码。
    

### 4. SiLU 激活函数 (SwiGLU 的基础)

-   **机制**：替代传统的 ReLU。数学公式为：
    
    $$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$
    
-   **动机**：ReLU 对负数一刀切直接截断为 $0$。SiLU 是一条平滑的曲线，在负数区域保留了一定的负值梯度，求导过程非常连续圆润，能大幅提升模型的各项考试成绩与表达性能。
    

## 二、 LLaMA 2 关键技术升级

```
                  ┌─► 1. 完整对齐流程 (SFT + RLHF 打造原生 Chat 版本)
                  ├─► 2. Ghost Attention (GAtt，死死记住初始上下文规矩)
[ LLaMA 2 四大升级 ] ┼─► 3. 2T 数据与高质量清洗 (Token 量增加 40%)
                  └─► 4. 4096 上下文窗口 + GQA 显存优化

```

### 1. 从基座模型到 Chat 对话助手

-   **LLaMA 1**：仅开源第一阶段预训练基座模型，需依靠开源社区微调（如 Alpaca）。
    
-   **LLaMA 2**：官方亲自走完 **SFT (监督微调)** 与 **RLHF (人类反馈强化学习)** 全流程，直接发布了安全性和情商更高的 `LLaMA-2-Chat` 官方版本。
    

### 2. Ghost Attention (GAtt 幽灵注意力)

-   **解决痛点**：多轮对话中，大模型容易在数轮之后遗忘最初设定的 System Prompt 角色规范。
    
-   **机制**：在微调阶段将初始系统指令隐式地绑定在后续每一轮对话的 Attention 掩码与特征中，确保模型在极长多轮对话中始终守规矩。
    

### 3. 数据量与上下文倍增

-   **训练 Token 增长**：从 1.4T 提升至 **2T (2万亿) Tokens**，并进行了极其严格的隐私与质量过滤。
    
-   **上下文窗口 (Context Window)**：从 2048 翻倍扩展至 **4096 个 Tokens**。
    

### 4. GQA (Grouped-Query Attention) 分组查询注意力

```
 Multi-head (MHA)            Grouped-query (GQA)           Multi-query (MQA)
  [Q Q Q Q Q Q Q Q]           [Q Q Q Q Q Q Q Q]            [Q Q Q Q Q Q Q Q]
  │ │ │ │ │ │ │ │              └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘             └───┼───┘
  [K K K K K K K K]            [K   K   K   K]                  [   K   ]
  [V V V V V V V V]            [V   V   V   V]                  [   V   ]
  (显存占用极高)               (最佳折中：聪明又省显存)          (速度最快，能力受损)

```

-   **演进背景**：MHA（多头注意力）每个 Key/Value 都有独立显存；MQA（单头共享）大幅省显存但模型变笨。
    
-   **GQA 机制**：将 Query 头分组，组内共享同一套 Key 和 Value 向量。在保留 MHA 高表达能力的同时，成倍降低了推理阶段 KV Cache 的显存占用。
    

## 三、 LLaMA 3 训练数据与架构微调

```
                 ┌─► 1. 15T 预训练 Token (LLaMA 2 的 7 倍，含 5% 高质量非英语)
                 ├─► 2. 词表扩充 4 倍 (32,000 ─► 128,000 Tokens)
[ LLaMA 3 核心亮点 ] ┼─► 3. 全系标配 GQA (8B 小模型也全面支持)
                 └─► 4. 上下文扩展至 8192 (8k)

```

1.  **海量高质数据**：预训练数据高达 **15T+ Tokens**（LLaMA 2 的 7 倍），代码数据增加了 4 倍，且包含 30 多种语言的非英语数据。微调阶段人工标注了 1000 万个样本。
    
2.  **128k 大词表（Tokenizer 升级）**：词表由 32,000 扩充 4 倍至 **128,000 (128k)**。一个中文汉字不再需要被切分为多个 Token，大幅提升了中文编码吞吐率与推理效率。
    
3.  **全系 GQA 覆盖**：连同 **8B 参数小模型** 也全面接入了 GQA 机制，进一步降低了端侧部署的显存门槛。
    
4.  **上下文翻倍**：序列长度（Context Length）进一步提升至 **8192 (8k)**。
    

## 四、 LLaMA 4 MoE 架构核心技术与前沿突破

LLaMA 4 走向了基于 Mixture-of-Experts (MoE) 的混合专家架构：

```
                            ┌─► 1. 稀疏激活 (Sparse Activation: 以 Maverick 为例，4000亿参数仅激活 170亿)
                            ├─► 2. 交错式设计 (Interleaved MoE/Dense Layers: 稠密层与 MoE 层交替)
[ LLaMA 4 MoE 架构三大核心 ] ┼─► 3. 专家架构 (Shared + Routed Experts: 共享专家 + 路由专家)
                            ├─► 4. 原生多模态 (Early Fusion: 视觉/文本 Token 从底层共同输入)
                            └─► 5. 千万级上下文 (iRoPE 架构支持 10M Context)

```

### 1. 极简“稀疏激活”（Sparse Activation）

-   **代表形态 (如 Maverick)**：拥有 4000 亿（400B）的总参数量，但单次前向传播仅激活 **170 亿（17B）** 参数（占比仅约 4.25%）。
    
-   **优势**：打破了传统 Scaling Laws 的算力瓶颈，既拥有千亿大模型的知识储备量，又具备百亿小模型的推理速度与超低计算延迟。
    

### 2. 交错式设计（Interleaved MoE/Dense Layers）

-   **机制**：并非所有层都是 MoE。LLaMA 4 采用了交错式设计，将传统的稠密层（Dense Layers）与 MoE 层交织排列。
    
-   **动机**：稠密层保证基础语义与常识的高效传递，MoE 层通过 Routing 分发解决专业知识的路由。
    

### 3. “共享 + 路由”专家机制（Shared + Routed Experts）

在多达 128 个专家的节点中，采用了创新的双层专家设计：

-   **共享专家（Shared Expert）**：常驻激活，专门处理通用语言知识和基础逻辑。
    
-   **路由专家（Routed Experts）**：由 Router 门控网络根据 Token 的特征（如遇到代码、数学公式等）动态挑选最合适的 1~2 个专家进行专门处理。
    
-   **节点优化**：专家节点内部使用经过极致硬件优化的 **SwiGLU 专家节点**。
    

### 4. MoE 带来的两大附带“神技”

-   **原生多模态（Early Fusion，早期融合）**：
    
    过去多模态通常是“先看图，翻译成文本，再交给大模型”。LLaMA 4 采用 Early Fusion，视觉 Token 和文本 Token 从最底层开始就直接送入同一个 MoE 骨干网络中，实现真正的跨模态统一理解。
    
-   **iRoPE 与千万级上下文 (10M Context)**：
    
    引入全新的 **iRoPE 架构**（交错旋转位置编码 + 拼接注意力机制），将上下文窗口直接爆改拉升到了 **10,000,000 (10M)** Tokens，可以直接一次性吞下整个大型 GitHub 代码库。