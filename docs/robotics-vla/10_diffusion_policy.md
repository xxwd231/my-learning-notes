
# Diffusion Policy 与 DDPM 变体深度解析精讲笔记
[📄 点击下载/预览笔记PDF](../../assets/10_Diffusion_Policy.pdf)
## 目录

-   [一、 策略生成分类与 Diffusion Policy 优势](https://www.google.com/search?q=%23%E4%B8%80-%E7%AD%96%E7%95%A5%E7%94%9F%E6%88%90%E5%88%86%E7%B1%BB%E4%B8%8E-diffusion-policy-%E4%BC%98%E5%8A%BF)
    
-   [二、 DDPM 向 Diffusion Policy 的改写机制](https://www.google.com/search?q=%23%E4%BA%8C-ddpm-%E5%90%91-diffusion-policy-%E7%9A%84%E6%94%B9%E5%86%99%E6%9C%BA%E5%88%B6)
    
-   [三、 闭环动作序列预测 (Close-loop Action Sequence Prediction)](https://www.google.com/search?q=%23%E4%B8%89-%E9%97%AD%E7%8E%AF%E5%8A%A8%E4%BD%9C%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B-close-loop-action-sequence-prediction)
    
-   [四、 神经网络架构：CNN-based vs. Transformer-based](https://www.google.com/search?q=%23%E5%9B%9B-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84cnn-based-vs-transformer-based)
    
-   [五、 Diffusion Policy 核心性质与训练稳定性](https://www.google.com/search?q=%23%E4%BA%94-diffusion-policy-%E6%A0%B8%E5%BF%83%E6%80%A7%E8%B4%A8%E4%B8%8E%E8%AE%AD%E7%BB%83%E7%A8%B3%E5%AE%9A%E6%80%A7)
    
-   [六、 关键实现细节与 DDPM/iDDPM/DDIM 变体对比](https://www.google.com/search?q=%23%E5%85%AD-%E5%85%B3%E9%94%AE%E5%AE%9E%E7%8E%B0%E7%BB%86%E8%8A%82%E4%B8%8E-ddpmiddpmddim-%E5%8F%98%E4%BD%93%E5%AF%B9%E6%AF%94)
    

## 一、 策略生成分类与 Diffusion Policy 优势

在模仿学习（Imitation Learning）与策略生成领域，常见策略构建方式可分为以下三类：

-   **显式策略 (Explicit Policy)**
    
    -   **工作机制**：直接将观测（Observation）映射到单点动作（Action）。
        
    -   **痛点局限**：精度较差，且无法解决多峰性（Multimodality）问题。
        
-   **隐式策略 (Implicit Policy)**
    
    -   **工作机制**：基于观测和动作学习能量函数，通过优化求解目标动作 $\arg\min_\mathbf{a} E(\mathbf{o}, \mathbf{a})$。
        
    -   **痛点局限**：无法解决负采样（Negative Sampling）准确性问题。
        
-   **Diffusion Policy**
    
    -   **工作机制**：端到端的训练，直接建模动作分布的梯度场（Gradient Field）。
        
    -   **核心优势**：
        
        -   **表达任意分布**：能表示任意归一化分布，完美解决多峰性问题。
            
        -   **长序列规划**：利用 Autoregressive 性质，支持在高维空间中进行长序列规划。
            
        -   **规避负采样**：直接从目标分布中学习，规避了非目标分布中的负采样问题，保证训练极度稳定。
            
        -   **融合控制思想**：结合了后退时域控制（Receding-Horizon Control，类似 MPC）概念，实现闭环且实时的动作序列生成。
            
        -   **条件分布解耦**：视觉变成了生成条件的一部分，而非输入的联合分布。
            

## 二、 DDPM 向 Diffusion Policy 的改写机制

Diffusion Policy 将图像生成中的 DDPM（去噪扩散概率模型）改写为针对机器人动作序列的条件生成模型：

### 核心公式对比

-   **DDPM（图像生成）**：
    
    $$\mathbf{x}^{k-1} = \alpha \left( \mathbf{x}^k - \gamma \boldsymbol{\epsilon}_\theta(\mathbf{x}^k, k) + \mathcal{N}(0, \sigma^2 \mathbf{I}) \right)$$
    
    $$\mathcal{L} = \text{MSE}\left(\boldsymbol{\epsilon}^k, \boldsymbol{\epsilon}_\theta(\mathbf{x}^0 + \boldsymbol{\epsilon}^k, k)\right)$$
    
-   **Diffusion Policy（动作生成）**：
    
    $$\mathbf{A}_t^{k-1} = \alpha \left( \mathbf{A}_t^k - \gamma \boldsymbol{\epsilon}_\theta(\mathbf{O}_t, \mathbf{A}_t^k, k) + \mathcal{N}(0, \sigma^2 \mathbf{I}) \right)$$
    
    $$\mathcal{L} = \text{MSE}\left(\boldsymbol{\epsilon}^k, \boldsymbol{\epsilon}_\theta(\mathbf{O}_t, \mathbf{A}_t^0 + \boldsymbol{\epsilon}^k, k)\right)$$
    

### 改写本质

1.  **输出替换**：生成目标从“图像”变为了“机器人的动作序列 $\mathbf{A}_t$”。
    
2.  **条件输入**：在去噪迭代过程中，将环境观测 $\mathbf{O}_t$ 作为条件（Condition）输入到去噪网络中。
    
3.  **条件分布本质**：观测数据不是联合分布，而是作为条件分布参与推理。
    

## 三、 闭环动作序列预测 (Close-loop Action Sequence Prediction)

Diffusion Policy 融入了类似滚动时域优化（MPC）的概念，以实现实时闭环控制：

```
[输入历史观测 O_t] 
       │
       ▼
[Diffusion Policy 预测动作序列 (Prediction Horizon T_p)] 
       │  (例如预测未来的 T_p 个动作: a_t, a_{t+1}, ..., a_{t+k})
       ▼
[仅执行第一个动作: a_t] 
       │
       ▼
[获得新的观测 O_{t+1}] ---> (周而复始)

```

-   **预测阶段**：输入一段历史观测 $\mathbf{O}_t$，模型预测出一整段动作序列 $\mathbf{A}_t$（预测时域 $T_p$）。
    
-   **执行阶段**：仅实际执行预测序列中的第一个动作 $a_{t}$，随后接收新观测重新预测，周而复始，兼顾了动作一致性与抗延迟能力。
    

## 四、 神经网络架构：CNN-based vs. Transformer-based

Diffusion Policy 支持搭建 CNN 和 Transformer 两种主干网络：

-   **CNN-based 架构**
    
    -   **结构细节**：采用一维时序 CNN（1D Temporal CNN），使用 **FiLM (Feature-wise Linear Modulation)** 进行特征嵌入与条件融合。
        
    -   **优点**：相对来说超参数非常稳定。
        
    -   **缺点**：在高频动作剧烈变化下，效果表现欠佳。
        
-   **Transformer-based 架构**
    
    -   **结构细节**：带噪动作作为 Token，去噪迭代次数作为第一个 Token；观测 $\mathbf{O}_t$ 用 MLP 转化为特征输入，并通过 **Cross Attention** 融合特征。
        
    -   **优点**：在复杂任务和高频动作生成下效果很好。
        
    -   **缺点**：很难训练。
        

## 五、 Diffusion Policy 核心性质与训练稳定性

### 1. 动作的多峰性（Multimodality）

Diffusion Model 的随机采样过程和随机初始化会导致动作的多峰化，通过随机优化方法，让生成结果可以收敛到不同的动作模态上。

### 2. 位置控制 vs 速度控制

-   **位置控制**：具有更好的多模态性质，能够更好地预测动作序列。
    
-   **速度控制**：更容易被复合误差（Compound Errors）影响。
    

### 3. 动作生成的关键维度

-   **一致性**：前后动作是有联系的动作序列。
    
-   **平稳性**：动作生成是丝滑的，尤其是静止动作（例如液体倾倒）。
    
-   **快速反应**：对环境变化具有较强的鲁棒性。
    

### 4. 训练稳定性机制（对比隐式策略 IBC）

-   **隐式策略 (IBC)** 依赖于基于能量的模型（EBM），其概率分布表示为 $p_\theta(\mathbf{a}\vert{}\mathbf{o}) = \frac{e^{-E_\theta(\mathbf{o}, \mathbf{a})}}{Z(\mathbf{o}, \theta)}$，通过 InfoNCE 进行负采样。负采样的准确性依赖于采样质量，若负样本无法充分覆盖动作空间，会导致归一化常数估计偏差，进而引发训练极其不稳定。
    
-   **Diffusion Policy** 直接求导建模动作梯度（Score Matching）：
    
    $$\nabla_\mathbf{a} \log p(\mathbf{a}\vert{}\mathbf{o}) = -\nabla_\mathbf{a} E_\theta(\mathbf{a}, \mathbf{o}) - \underbrace{\nabla_\mathbf{a} \log Z(\mathbf{o}, \theta)}_{=0} \approx -\boldsymbol{\epsilon}_\theta(\mathbf{a}, \mathbf{o})$$
    
    公式中导数成功清零消除了配分函数 $Z$，彻底消除了负采样问题，极大地保证了训练稳定性。
    

## 六、 关键实现细节与 DDPM/iDDPM/DDIM 变体对比

### 1. 实机训练细节

-   **多相机输入**：训练时使用不止一个相机（例如前置相机与手腕相机），图像跟着时间戳单独编码，一个用来保留深度信息，另一个用来保证训练稳定。
    
-   **Noise Schedule 优化**：试了 DDPM 的线性方法发现不好用，改用 iDDPM 的 Cosine 余弦方法来实现。
    
-   **采样加速**：用 DDIM 的方法来实现正向过程与反向过程的解耦，推理速度能做到在 RTX 3080 上迭代 100 次只要 0.1 秒。
    

### 2. 三大 Diffusion 变体对比详解

-   **DDPM 原版**
    
    -   **改动对象**：完整基础框架。
        
    -   **噪声调度**：Linear 线性。
        
    -   **反向方差**：固定不可学。
        
    -   **采样步数**：必须 1000 步，速度很慢。
        
    -   **随机性**：每步必加噪声，随机生成。
        
    -   **与 DP 关系**：原生调度效果差，在 Diffusion Policy 中被弃用。
        
-   **iDDPM (Improved DDPM)**
    
    -   **改动对象**：训练 + 正向噪声调度。
        
    -   **噪声调度**：Cosine 余弦（**DP 训练选用**）。
        
    -   **反向方差**：网络可学习方差。
        
    -   **采样步数**：同 DDPM 原生采样，速度较慢。
        
    -   **随机性**：原生采样仍具随机性。
        
    -   **与 DP 关系**：**DP 训练标准方案**。
        
-   **DDIM**
    
    -   **改动对象**：推理采样算法。
        
    -   **噪声调度**：复用训练时任意 schedule。
        
    -   **反向方差**：方差为可调超参 $\eta$。
        
    -   **采样步数**：**可压缩至 10~100 步，实现极速采样**。
        
    -   **随机性**：当 $\eta=0$ 时可做到完全确定。
        
    -   **与 DP 关系**：**DP 推理加速必备**，其训练权重能完美适配并兼容 DDPM/iDDPM 训练出的网络。