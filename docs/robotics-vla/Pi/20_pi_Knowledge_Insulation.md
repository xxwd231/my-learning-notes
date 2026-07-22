
# 知识绝缘 (Knowledge Insulation) VLA 架构精讲笔记
[📄 点击下载/预览笔记PDF](../../assets/20_pi_Knowledge_Insulation VLA.pdf)
## 目录

-   [一、 核心问题诊断与知识绝缘技术](https://www.google.com/search?q=%23%E4%B8%80-%E6%A0%B8%E5%BF%83%E9%97%AE%E9%A2%98%E8%AF%8A%E6%96%AD%E4%B8%8E%E7%9F%A5%E8%AF%86%E7%BB%9D%E7%BC%98%E6%8A%80%E6%9C%AF)
    
-   [二、 解决方案总览：三招合一](https://www.google.com/search?q=%23%E4%BA%8C-%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88%E6%80%BB%E8%A7%88%E4%B8%89%E6%8B%9B%E5%90%88%E4%B8%80)
    
-   [三、 联合训练架构与损失函数](https://www.google.com/search?q=%23%E4%B8%89-%E8%81%94%E5%90%88%E8%AE%AD%E7%BB%83%E6%9E%B6%E6%9E%84%E4%B8%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0)
    
-   [四、 主干混合训练数据分布](https://www.google.com/search?q=%23%E5%9B%9B-%E4%B8%BB%E5%B9%B2%E6%B7%B7%E5%90%88%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE%E5%88%86%E5%B8%83)
    
-   [五、 训练时 vs 推理时信息流向](https://www.google.com/search?q=%23%E4%BA%94-%E8%AE%AD%E7%BB%83%E6%97%B6-vs-%E6%8E%A8%E7%90%86%E6%97%B6%E4%BF%A1%E6%81%AF%E6%B5%81%E5%90%91)
    
-   [六、 $\pi_{0.5}$ 与知识绝缘方案对比](https://www.google.com/search?q=%23%E5%85%AD-%CF%8005-%E4%B8%8E%E7%9F%A5%E8%AF%86%E7%BB%9D%E7%BC%98%E6%96%B9%E6%A1%88%E5%AF%B9%E6%AF%94)
    

## 一、 核心问题诊断与知识绝缘技术

-   **问题诊断**[cite: 10]：
    
    -   在传统的 Flow Matching VLA 模型中，**随机初始化的 Action Expert** 在训练初期会产生巨大且无序的梯度回传[cite: 10]。
        
    -   这些噪音梯度会严重干扰甚至破坏 VLM 主干（Backbone）原本在预训练阶段学到的通用知识，进而影响语言跟随能力与整体收敛速度[cite: 10]。
        
-   **知识绝缘技术 (Knowledge Insulation)**[cite: 10]：
    
    -   通过精确设置 **stop-gradient (梯度截断)**，彻底切断从 Action Expert 反向传播到 VLM Backbone 的梯度流[cite: 10]。
        
    -   同时保留前向传播的信息传递，确保 Action Expert 依然可以正常读取 VLM Backbone 提炼出的高层特征[cite: 10]。
        

## 二、 解决方案总览：三招合一

方案通过整合三项核心技术，兼顾了**训练快、推理快、泛化好**三个维度[cite: 10]：

-   **1. 联合训练 (Joint Training)**[cite: 10]
    
    -   **解决问题**：解决 Action Expert 训练初期无法为 Backbone 提供优质学习信号的问题[cite: 10]。
        
    -   **达成效果**：利用离散 FAST token 的自回归损失（AR loss）为 VLM Backbone 提供干净、高效的学习梯度[cite: 10]。
        
-   **2. 知识绝缘 (Knowledge Insulation)**[cite: 10]
    
    -   **解决问题**：解决随机初始化的 Action Expert 带来的梯度污染问题[cite: 10]。
        
    -   **达成效果**：使用 stop-gradient 停止 Action Expert 到 VLM Backbone 的梯度更新，保护 VLM 预训练知识[cite: 10]。
        
-   **3. VLM 数据协同训练 (Co-Training)**[cite: 10]
    
    -   **解决问题**：解决在机器人控制微调过程中可能出现的预训练知识遗忘问题[cite: 10]。
        
    -   **达成效果**：混合海量通用 VLM 数据，持续巩固并保持大模型的语言理解与视觉推理能力[cite: 10]。
        

## 三、 联合训练架构与损失函数

两种动作表示在同一个模型中共存，在**同一次 forward (前向传播)** 中同时计算两个 loss[cite: 10]：

-   **联合损失函数公式**[cite: 10]：
    
    $$\mathcal{L} = - \sum_{j} M^l_j \log p_\theta(\hat{\ell}_{j+1} \mid x_{1:j}) + \alpha M^{\text{act}} \Vert{}\dots\Vert{}^2$$
    
-   **损失项拆解**[cite: 10]：
    
    -   **前项（AR loss）**：FAST离散 token 交叉熵损失，仅用于更新 VLM Backbone（3B），作为训练辅助[cite: 10]。
        
    -   **后项（Flow Matching loss）**：连续动作预测损失，仅用于更新 Action Expert（300M）[cite: 10]。
        
-   **独立优化与 Mask 机制**[cite: 10]：
    
    -   依靠 stop-gradient 保证，两个 loss 的参数更新完全独立，直接设置 $\alpha = 1$ 即可，无需复杂调参[cite: 10]。
        
    -   通过 Attention Mask 机制，让 FAST token 与连续动作 token 在注意力计算中互相屏蔽[cite: 10]。
        

## 四、 主干混合训练数据分布

VLM 主干作为“多任务学习者”，而 Action Expert 作为专精控制的“专家模块”[cite: 10]：

-   **1. 机器人动作数据（图像 + 指令 + 轨迹）**[cite: 10]
    
    -   **主干策略**：执行 FAST token 预测[cite: 10]。
        
    -   **Action Expert 策略**：执行 Flow Matching 控制生成[cite: 10]。
        
-   **2. 通用 VLM 数据（VQA、图像描述、目标检测）**[cite: 10]
    
    -   **主干策略**：执行标准语言 token 预测[cite: 10]。
        
    -   **Action Expert 策略**：**不参与计算**（设置 $M^{\text{act}} = 0$，不计算动作 loss）[cite: 10]。
        
-   **3. 机器人规划数据（机器人数据 + 下一步语言标注）**[cite: 10]
    
    -   **主干策略**：执行语言 token 预测[cite: 10]。
        
    -   **Action Expert 策略**：执行 Flow Matching 控制生成[cite: 10]。
        
-   **数据协同效果**[cite: 10]：
    
    -   语言知识促进动作理解，机器人数据建立控制表示，二者相互促进；Action Expert 则专注于“把好特征转化为精确连续动作”[cite: 10]。
        

## 五、 训练时 vs 推理时信息流向

### 1. 训练阶段 (Training)

-   **输入**：[图像] [文字] [状态][cite: 10]。
    
-   **双分支并行**[cite: 10]：
    
    -   **分支 A**：VLM Backbone 预测 FAST token 并计算 AR loss（仅更新 Backbone）[cite: 10]。
        
    -   **分支 B**：VLM Backbone 提取特征传递给 Action Expert，**中间通过 `sg` (stop-gradient) 截断梯度**，Action Expert 计算 Flow Matching loss（仅更新 Action Expert）[cite: 10]。
        

### 2. 推理阶段 (Inference)

-   **输入**：[图像] [文字] [状态][cite: 10]。
    
-   **单路径快速输出**[cite: 10]：
    
    -   **FAST token 路径完全关闭**，不参与推理生成[cite: 10]。
        
    -   VLM Backbone 提取高层特征后，直接送入 Action Expert，由 Flow Matching 以极高速度输出连续动作轨迹（10Hz 控制）[cite: 10]。
        

## 六、 $\pi_{0.5}$ 与知识绝缘方案对比

-   **训练阶段划分**[cite: 10]：
    
    -   **$\pi_{0.5}$**：采用**两阶段训练**（先进行离散 FAST token 预训练，后续阶段再加入 Action Expert 联合训练）[cite: 9, 10]。
        
    -   **知识绝缘方案**：采用**单阶段训练**（通过 stop-gradient 直接替代阶段隔离，实现同时训练）[cite: 10]。
        
-   **核心优势**[cite: 10]：
    
    -   第一次真正实现了训练快（FAST 高散 token）+ 推理快（Flow Matching Expert）+ 泛化好（VLM 数据协同）三者兼得的完美 VLA 方案[cite: 10]。