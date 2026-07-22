
# Action Chunking with Transformers (ACT) 深度解析与精讲笔记
[📄 点击下载/预览笔记PDF](../../assets/07_ACT模型.pdf)
## 目录

-   [一、 核心动机与背景概念](https://www.google.com/search?q=%23%E4%B8%80-%E6%A0%B8%E5%BF%83%E5%8A%A8%E6%9C%BA%E4%B8%8E%E8%83%8C%E6%99%AF%E6%A6%82%E5%BF%B5)
    
-   [二、 为什么需要动作分块（Why Action Chunking?）](https://www.google.com/search?q=%23%E4%BA%8C-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E5%8A%A8%E4%BD%9C%E5%88%86%E5%9D%97why-action-chunking)
    
-   [三、 时间集成（Temporal Ensembling）](https://www.google.com/search?q=%23%E4%B8%89-%E6%97%B6%E9%97%B4%E9%9B%86%E6%88%90temporal-ensembling)
    
-   [四、 ACT 算法原理与损失函数](https://www.google.com/search?q=%23%E5%9B%9B-act-%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86%E4%B8%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0)
    
-   [五、 ACT 网络架构与训练/测试流程](https://www.google.com/search?q=%23%E4%BA%94-act-%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84%E4%B8%8E%E8%AE%AD%E7%BB%83%E6%B5%8B%E8%AF%95%E6%B5%81%E7%A8%8B)
    
-   [六、 ACT 模型核心总结](https://www.google.com/search?q=%23%E5%85%AD-act-%E6%A8%A1%E5%9E%8B%E6%A0%B8%E5%BF%83%E6%80%BB%E7%BB%93)
    

## 一、 核心动机与背景概念

在机器人模仿学习（Imitation Learning）中，传统单步策略（Single-Step Policy）容易累积误差并忽略时间依赖关系。**ACT (Action Chunking with Transformers)** 通过结合 **CVAE（条件变分自编码器）** 与 **Transformer** 架构，将未来的连续 $k$ 步动作作为一个动作块（Action Chunk）进行预测，有效解决了多模态动作分布建模与长时间跨度操作中的误差累积问题。

## 二、 为什么需要动作分块（Why Action Chunking?）

### 1. 复合误差问题（Compounding Errors）

-   单步策略 $\pi_\theta(a_t \vert{} s_t)$ 在早期步骤中产生的微小预测误差随着时间推进不断累积，最终导致长期决策严重偏离正确轨迹。
    
-   动作分块策略 $\pi_\theta(a_{t:t+k} \vert{} s_t)$ 一次预测未来的 $k$ 步动作（$k$ timesteps, one decision made），有效缩短了长轨迹的有效时间跨度（Reduce the effective horizon），大幅降低犯错概率。
    

### 2. 非马尔可夫行为与时间依赖（Non-Markov Performance）

-   传统 RL/IL 假设未来状态仅依赖于当前状态 $P(a_t \vert{} s_t, s_{t-1}, \dots) = P(a_t \vert{} s_t)$。然而真实任务中状态空间无法包含所有历史信息（如“炒菜时暂停 3 秒等待油热”）。
    
-   动作分块在单个 Chunk 内建模时间依赖关系（如“倒油，等待 3 秒，然后……”）。
    

### 3. 避免因果误判（Casual Misidentification）

-   直接将历史状态输入模型 $\pi_\theta(a_t \vert{} s_t, s_{t-1}, \dots)$ 极易引发因果误判。
    
-   在自动驾驶场景中，模型可能会错误地以为“刹车灯亮起”是“踩刹车”的原因（准确率 90%），而不是因为“路面有障碍物”（准确率 100%）。由于历史 context 越长这种误判越严重，ACT 选择通过输出未来的 Action Chunk 来显式建模时间依赖，避免了对漫长历史状态的过度依赖。
    

## 三、 时间集成（Temporal Ensembling）

为了防止动作块在拼接切换时产生断续抖动（Jerky execution），ACT 提出了 **时间集成（Temporal Ensembling）** 平滑机制：

-   **机制**：在每个时间步 $t$，模型都会输出一个包含未来 $k$ 步动作的 Chunk。因此，对于某一个具体的执行时间点，会有多个来自不同预测时间步的动作重叠。
    
-   **加权平均**：使用指数衰减权重对这些重叠的预测动作进行加权平均计算：
    
    $$w_i = \exp(-m \cdot i)$$
    
    （其中 $i$ 表示该预测动作在其所在的 Chunk 中的相对延时/位置）。
    
-   **效果**：极大地平滑了执行端的动作输出，消除了机械臂运动中的卡顿与抖动现象。
    

## 四、 ACT 算法原理与损失函数

ACT 将模仿学习构建为一个 CVAE 模型：

-   **编码器（CVAE Encoder）**：仅在**训练阶段**使用，输入当前观察 $o_t$ 与地面真实动作序列 $a_{t:t+k}$，输出概率分布参数 $\mu$ 与 $\sigma^2$。
    
-   **解码器（CVAE Decoder）**：结合当前观察 $o_t$ 与隐变量 $z$，预测未来动作序列 $\hat{a}_{t:t+k}$。
    

### 1. 损失函数

ACT 的训练损失由两部分组成：

$$\mathcal{L}_{\text{ACT}} = \mathcal{L}_{\text{reconstruct}} + \beta \mathcal{L}_{\text{reg}}$$

-   **重构损失 ($\mathcal{L}_{\text{reconstruct}}$)**：对齐预测动作与真实动作（通常采用 MSE Loss）：
    
    $$\mathcal{L}_{\text{reconstruct}} = \text{MSE}(\hat{a}_{t:t+k}, a_{t:t+k})$$
    
-   **正则化项 ($\mathcal{L}_{\text{reg}}$)**：强制隐分布接近标准高斯分布 $N(0, \mathbf{I})$：
    
    $$\mathcal{L}_{\text{reg}} = D_{\text{KL}}(q_\phi(z \vert{} a_{t:t+k}, \bar{o}_t) \parallel N(0, \mathbf{I}))$$
    
    其闭式解展开为：
    
    $$D_{\text{KL}} = \frac{1}{2} \sum_{i=1}^{d} \left( 1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2 \right)$$
    

## 五、 ACT 网络架构与训练/测试流程

### 1. 训练阶段（Training）

1.  **数据采样**：从 Demonstration 数据集中采样当前观察 $o_t$（包括 4 路 RGB 相机图像与从臂关节角度 `joints`）以及真实动作序列 $a_{t:t+k}$。
    
2.  **推导隐变量 $z$**：
    
    -   `[CLS]` Token、`joints` 关节向量与 $a_{t:t+k}$ 动作序列输入 Transformer Encoder（4x self-attention blocks）。
        
    -   提取 `[CLS]` 位置的输出特征映射为 $\mu$ 与 $\log(\sigma^2)$。
        
    -   通过重参数化技巧 $z = \mu + \sigma \cdot \epsilon$（$\epsilon \sim N(0, \mathbf{I})$）得到隐变量 $z$。
        
3.  **预测动作序列**：
    
    -   4 路 RGB 图像经过 ResNet-18 提取特征后展平，与位置编码融合（生成 $300 \times 4 = 1200$ 个视觉 Token）。
        
    -   视觉 Token、`joints` 嵌入向量以及隐变量 $z$ 组合输入主 Transformer Encoder（4x self-attention blocks）。
        
    -   主 Encoder 的输出作为 Cross-Attention 键值（K, V）送入 Transformer Decoder（7x cross-attention blocks），解码出未来的预测动作序列 $\hat{a}_{t:t+k}$。
        

### 2. 测试/推理阶段（Testing）

-   **丢弃 CVAE Encoder**：在推理时完全丢弃编码器，不计算隐分布。
    
-   **隐变量设为零**：直接将隐变量 $z$ 设置为零向量（$z = 0$）。
    
-   **端到端生成**：仅依靠当前的视觉观察与从臂关节角度输入 Transformer 主干，直接解码预测动作序列并使用 Temporal Ensembling 执行。
    

## 六、 ACT 模型核心总结

-   **时序分块解决误差**：通过 Action Chunking 一次性预测 $k$ 步动作，极大地缓解了模仿学习中的复合误差与因果误判问题。
    
-   **表达多模态轨迹**：利用 CVAE 隐空间建模同一个观察状态下可能存在的多种合法动作轨迹。
    
-   **平滑执行**：Temporal Ensembling 机制通过跨时间步加权平均，确保了真实机械臂控制的高平滑度。
    
-   **局限性**：单个 ACT 模型通常只针对特定单一任务训练，缺乏对跨场景、跨任务的泛化能力（一个任务对应一个模型）。