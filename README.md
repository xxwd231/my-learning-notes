# 📚 Embodied AI & VLA Paper Notes (具身智能与多模态大模型精讲笔记)

本项目收录了从 **大语言模型 (LLM)**、**多模态视觉语言模型 (VLM)** 到 **具身智能与视觉-语言-动作模型 (VLA / Robotics)** 的系列经典与前沿论文精读笔记。

笔记全面采用规范的层级列表与公式解析，旨在清晰呈现模型架构、核心洞察、损失函数及训练/推理逻辑，方便快速查阅与系统学习。

---

## 🛠️ 项目结构 (Project Structure)

仓库主要由两个核心目录构成：
* `docs/`：包含所有 Markdown 格式的论文精讲笔记，按技术演进分类整理。
* `assets/`：存放对应论文的原始 PDF 资料及相关架构图解。

```text
my-paper-notes/
├── assets/                  # 论文 PDF 原件与图片资源
│   ├── 01_transformer.pdf
│   ├── 02_bert论文.pdf
│   └── ... (01~23 论文 PDF)
├── docs/                    # 论文精读笔记 (Markdown)
│   ├── llm-nlp/             # 语言大模型基础
│   ├── multimodal-vlm/      # 多模态视觉大模型
│   └── robotics-vla/        # 具身智能与机器人控制 (RT, Pi等)
└── README.md                # 本文档



## 📖 笔记导航与技术演进 (Table of Contents)

### 1. 语言与基础大模型 (LLM & Foundation)

-   **01_Transformer** - Attention Is All You Need (自注意力机制与 Transformer 架构)
    
-   **02_BERT** - 双向 Encoder 预训练语言模型
    
-   **06_VAE** - 变分自编码器与生成模型基础
    
-   **14_LLaMA** - 开源大语言模型基座与高效架构
    

### 2. 多模态视觉语言模型 (Multimodal & VLM)

-   **04_ViT** - Vision Transformer (视觉 Token 化与 Transformer 图像处理)
    
-   **08_CLIP** - 基于对比学习的双塔图文跨模态表征
    
-   **05_BLIP-2** - Q-Former 桥接冻结视觉编码器与 LLM
    
-   **03_LLaVA** - 视觉指令微调与大型多模态模型
    

### 3. 具身智能与机器人策略 (Robotics & VLA)

#### 基础动作策略与模拟控制

-   **07_ACT** - Action Chunking with Transformers (动作块与 CVAE 模仿学习)
    
-   **10_Diffusion Policy** - 基于扩散模型的机器人连续轨迹生成策略
    

#### 谷歌 RT (Robotics Transformer) 系列

-   **11_RT-1** - 基于 Transformer 的实时机器人控制网络
    
-   **12_RT-2** - 将 VLM 表现力转化为机器人动作控制 (Vision-Language-Action)
    
-   **13_RT-X** - 跨机器人硬件平台的大规模具身数据集与通用策略
    

#### 轻量化与高效开源 VLA

-   **09_SmolVLA** - 轻量化具身端侧部署模型
    
-   **15_OpenVLA** - 开源可微调的大型视觉-语言-动作模型
    

#### Physical Intelligence ($\pi$) 系列及最新演进

-   **16_pi0** - Physical Intelligence 具身通用控制基座模型
    
-   **17_pi0-FAST** - 离散自回归 Token 化的高效动作生成机制
    
-   **18_pi_Hi_Robot** - 高动态与高精细度双臂/多任务机器人控制
    
-   **19_pi0.5** - 两阶段训练与连续/离散混合动作表示
    
-   **20_pi_Knowledge_Insulation** - 知识绝缘技术：解决 Action Expert 梯度污染 VLM 主干问题
    
-   **21_piRAC** - RTC (Real-Time Control) 实时控制与 Soft Masking Inpainting 延迟抹平机制
    
-   **22_pi0.6** - Recap 强化学习：利用独立 Value Model 解决长程任务信用分配 (Credit Assignment)
    
-   **23_pi0.7** - 多模态 Prompt、世界模型生成视觉子目标图像与 Dropout 机制
    

## 🎯 笔记核心特点

1.  **结构统一规范**：全篇采用层级列表组织，无冗余墙面文本，具备极高的可读性与可扫描性。
    
2.  **注重公式与推导**：关键 loss（如 Flow Matching loss, AR loss, Value Estimation 等）均采用标准 LaTeX 显示格式呈列。
    
3.  **直击核心痛点**：深入分析各模型的 **核心洞察 (Core Insights)**、**痛点解法**、**架构对比** 以及 **局限性**。
    

## 🤝 贡献与交流 (Contribution)

欢迎对具身智能（Embodied AI）、VLA 大模型、机器人模仿学习/强化学习感兴趣的朋友提 Issue 或 PR，共同补充与完善最新的具身大模型笔记！