
# BLIP-2 架构与原理精讲笔记
[📄 点击下载/预览笔记PDF](../../assets/05_BLIP_2.pdf)
## 目录

-   [一、 核心动机与整体架构](https://www.google.com/search?q=%23%E4%B8%80-%E6%A0%B8%E5%BF%83%E5%8A%A8%E6%9C%BA%E4%B8%8E%E6%95%B4%E4%BD%93%E6%9E%B6%E6%9E%84)
    
-   [二、 核心模块：Q-Former (Querying Transformer)](https://www.google.com/search?q=%23%E4%BA%8C-%E6%A0%B8%E5%BF%83%E6%A8%A1%E5%9D%97q-former-querying-transformer)
    
-   [三、 第一阶段：视觉-语言表示学习（三大训练目标）](https://www.google.com/search?q=%23%E4%B8%89-%E7%AC%AC%E4%B8%80%E9%98%B6%E6%AE%B5%E8%A7%86%E8%A7%89-%E8%AF%AD%E8%A8%80%E8%A1%A8%E7%A4%BA%E5%AD%A6%E4%B9%A0%E4%B8%89%E5%A4%A7%E8%AE%AD%E7%BB%83%E7%9B%AE%E6%A0%87)
    
-   [四、 第二阶段：视觉到语言生成预训练（连接 LLM）](https://www.google.com/search?q=%23%E5%9B%9B-%E7%AC%AC%E4%BA%8C%E9%98%B6%E6%AE%B5%E8%A7%86%E8%A7%89%E5%88%B0%E8%AF%AD%E8%A8%80%E7%94%9F%E6%88%90%E9%A2%84%E8%AE%AD%E7%BB%83%E8%BF%9E%E6%8E%A5-llm)
    
-   [五、 下游任务微调与推理（以 VQA 为例）](https://www.google.com/search?q=%23%E4%BA%94-%E4%B8%8B%E6%B8%B8%E4%BB%BB%E5%8A%A1%E5%BE%AE%E8%B0%83%E4%B8%8E%E6%8E%A8%E7%90%86%E4%BB%A5-vqa-%E4%B8%BA%E4%BE%8B)
    
-   [六、 核心优势与总结](https://www.google.com/search?q=%23%E5%85%AD-%E6%A0%B8%E5%BF%83%E4%BC%98%E5%8A%BF%E4%B8%8E%E6%80%BB%E7%BB%93)
    

## 一、 核心动机与整体架构

传统多模态预训练模型端到端训练成本高昂，且难以直接复用单模态领域最顶尖的视觉模型和大语言模型（LLM）。**BLIP-2** 提出了 **两阶段预训练策略**，使用轻量级的 **Q-Former** 作为桥梁，在冻结参数的前提下弥补视觉和文本的模态鸿沟（Modality Gap）。

Plaintext

```
[冻结的图像编码器 (ViT)] ──> [可训练的 Q-Former] ──> [线性投影/FC] ──> [冻结的大语言模型 (LLM)]

```

### 整体框架设计：

-   **冻结图像编码器（Frozen Image Encoder）**：如 EVA-CLIP / ViT-L，提取高质量图像特征。
    
-   **轻量级 Q-Former**：作为模态桥梁，从图像特征中提取与文本相关的固定数量语义表征。
    
-   **冻结大语言模型（Frozen LLM）**：如 OPT, Flan-T5，利用强大的语言生成与推理能力。
    

## 二、 核心模块：Q-Former (Querying Transformer)

Q-Former 是 BLIP-2 的核心创新结构，由两个共享 Self-Attention 层的子模块组成：

-   **图像 Transformer**：与图像编码器互动，通过 Cross-Attention 提取视觉特征。
    
-   **文本 Transformer**：既作为文本编码器，也作为文本解码器。
    

### Learned Queries（可学习查询向量）

-   **输入设计**：设置固定数量（如 $32$ 个）的可学习 Embedding 参数，维度与 Transformer 隐层维度一致（如 $d=768$）。
    
-   **作用**：不论输入图像分辨率多大、生成的图像 Token 序列多长（如 $257$ 个），通过 Cross-Attention 处理后均压缩为固定数量（$32$ 个）的视觉 Token，极大降低了后续送入 LLM 的计算开销。
    

## 三、 第一阶段：视觉-语言表示学习（三大训练目标）

在第一阶段，**Image Encoder 保持冻结**，仅训练 Q-Former。通过控制 **Self-Attention Mask** 策略，同时优化三个互补的训练目标：

### 1. 图像-文本对比学习 (ITC, Image-Text Contrastive)

-   **作用与机制**：对齐视觉特征与文本特征，最大化正样本对的互信息。提取 Query 的输出与文本 `[CLS]` 算余弦相似度。
    
-   **Attention Mask 策略**：**单模态掩码 (Uni-modal)**，Query 和 Text 互不干扰，各自做自注意力。
    

### 2. 基于图像的文本生成 (ITG, Image-Grounded Text Generation)

-   **作用与机制**：训练 Q-Former 在给定视觉特征约束下生成对应文本描述。
    
-   **Attention Mask 策略**：**多模态因果掩码 (Multimodal Causal)**，Query 可以互相注意；Text 只能注意 Query 和它之前的 Text。
    

### 3. 图像-文本匹配 (ITM, Image-Text Matching)

-   **作用与机制**：二分类任务（是/否匹配），学习图文细粒度的对齐关系。
    
-   **Attention Mask 策略**：**双向多模态掩码 (Bi-directional)**，Query 和 Text 之间可以无阻碍自由相互注意。
    

## 四、 第二阶段：视觉到语言生成预训练（连接 LLM）

第二阶段将 Q-Former 输出的视觉 Token 连接至 **冻结的 LLM**，通过 Fully Connected (FC) 线性层进行维度映射（映射至 LLM 对应的输入通道数）。

### 1. Bootstrapping Decoder-based LLM (如 OPT)

-   **机制**：Q-Former 输出的特征向量经 FC 变换后，作为 **软提示词（Soft Visual Prompts）** 拼接到 LLM 的文本输入序列最前面。
    
-   **损失**：采用标准的语言模型交叉熵损失（Causal Language Modeling Loss），预测后续文本。
    

### 2. Bootstrapping Encoder-Decoder LLM (如 Flan-T5)

-   **机制**：将视觉特征与 Prefix Text（如 Prompt 提示词）拼接后送入 LLM Encoder，LLM Decoder 负责预测目标 Suffix Text。
    

## 五、 下游任务微调与推理（以 VQA 为例）

在视觉问答（VQA）任务中，需要根据图像与问题（Question）预测答案（Answer）：

1.  **Condition 注入**：将用户提出的 Question 文本作为条件输入给 Q-Former。
    
2.  **针对性提取**：Q-Former 的 Query Token 通过 Cross-Attention 提取与该问题最相关的视觉特征。
    
3.  **LLM 生成答案**：经过 FC 映射后的视觉表征与 Question 共同作为输入送入 LLM，LLM 输出最终答案（如 `"sunglasses"`）。
    

## 六、 核心优势与总结

-   **训练效率极高**：相比从头训练或全量微调多模态模型，参数量仅调整 Q-Former 和 FC 层，显著降低算力开销。
    
-   **可解释与高效压缩**：通过 Query 机制将长图像序列浓缩为固定 $32$ 个高语义 Token，解耦了视觉通道数与 LLM 上下文长度。
    
-   **强大的 Zero-shot 能力**：通过适配成熟的指令微调 LLM（如 Flan-T5），直接具备了强大的 Zero-shot 图像到文本理解与对话能力。