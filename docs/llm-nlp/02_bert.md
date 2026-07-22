
# BERT深度原理与架构精讲笔记：双向 Transformer 语言模型
[📄 点击下载/预览笔记PDF](../../assets/02_bert论文.pdf)

## 前言

### 1. 技术背景

在自然语言处理（NLP）领域，预训练语言模型经历了从静态词向量（如 Word2Vec、GloVe）到动态上下文表示（如 ELMo、OpenAI GPT）的演进。传统的自回归语言模型（Autoregressive Language Models）通常使用单向架构（如从左到右），这种单向性严重限制了模型在需要同时依赖上下文信息的词级和句子级任务上的表现。

### 2. 核心问题

-   **单向上下文限制**：如 OpenAI GPT 仅采用从左到右的自注意力机制，在问答（QA）或序列标注等任务中，无法在同一层协同吸收左右双向的上下文信息。
    
-   **多任务泛化能力**：传统方法在迁移到下游任务时，往往需要针对特定任务设计复杂的手工架构（如 ELMo 结合 Task-specific LSTM）。
    

### 3. 适用对象

本笔记适用于从事深度学习、自然语言处理（NLP）、大语言模型（LLM）研发的工程人员、算法工程师及计算机专业研究人员。

## 目录

-   **1. BERT 核心原理与思想**
    
    -   1.1 什么是 BERT？
        
    -   1.2 双向上下文表征对比（BERT vs. GPT vs. ELMo）
        
    -   1.3 预训练 + 微调 Paradigm
        
-   **2. BERT 模型架构与输入表征**
    
    -   2.1 Transformer Encoder 架构细节
        
    -   2.2 统一输入表示 (Input Representation)
        
    -   2.3 特殊 Token 的作用 (`[CLS]`与`[SEP]`)
        
-   **3. 预训练任务（Pre-training Objectives）深度拆解**
    
    -   3.1 掩码语言模型（Masked Language Model, MLM）
        
    -   3.2 下句预测任务（Next Sentence Prediction, NSP）
        
    -   3.3 联合损失函数计算
        
-   **4. 下游任务微调范式（Fine-Tuning Strategies）**
    
    -   4.1 句对分类任务
        
    -   4.2 单句分类任务
        
    -   4.3 抽取式问答任务 (Question Answering)
        
    -   4.4 序列标注任务 (Sequence Tagging)
        
-   **5. 基于 Feature-based 的提取模式与消融实验**
    
    -   5.1 Fine-Tuning vs. Feature-Based
        
    -   5.2 模型规模与消融分析
        
-   **6. 总结与后续演进**
    

## 1. BERT 核心原理与思想

### 1.1 什么是 BERT？

BERT的全称是 **Bidirectional Encoder Representations from Transformers**。它是由 Devlin 等人（Google AI Language）于 2018 年提出的预训练语言模型。其核心突破在于首次真正实现了深层**双向**（Deep Bidirectional）Transformer Encoder 的无监督预训练。

### 1.2 双向上下文表征对比

在 BERT 出现之前，主流的上下文表示模型主要分为两类：

-   **OpenAI GPT (单向 Transformer)**：
    
    -   使用标准的从左到右 Transformer Decoder（带 Mask 的 Self-Attention）。
        
    -   每个 Token 只能注意到当前位置之前的 Token。
        
-   **ELMo (浅层双向 LSTM)**：
    
    -   分别独立训练一个从左到右的 LSTM和一个从右到左的 LSTM。
        
    -   将两者的隐藏层状态直接拼接（Concatenation）作为最终特征。
        
    -   并非真正的“深层联合双向” conditioning。
        
-   **BERT (深层深联合双向 Transformer)**：
    
    -   在 Transformer 的所有层中，联合基于左侧和右侧的上下文进行条件建模。
        
    -   解决了传统语言模型由于防止“信息泄漏”（即通过右侧上下文预测自身）而被迫采用单向架构的问题。
        

```
    BERT (双向)                 GPT (单向)                ELMo (浅层双向)
    
     T1  T2  T3                 T1  T2  T3               T1    T2    T3
      ^   ^   ^                  ^   ^   ^                ^     ^     ^
     [Trm-Encoder]              [Trm-Decoder]          [LSTM] [LSTM] [LSTM]
     / \ / \ / \                /   /   /             ( Left ) ( Right )
    E1  E2  E3                 E1  E2  E3                E1    E2    E3

```

### 1.3 预训练 + 微调范式

BERT 极大地简化了下游 NLP 任务的开发流程：

-   **预训练阶段 (Pre-training)**：在海量无标注文本（如 Wikipedia 和 BooksCorpus）上，利用自监督任务训练通用语言表示。
    
-   **微调阶段 (Fine-tuning)**：针对具体的下游任务，只需添加极少量的 Task-specific 输出层，并在下游数据上全量微调所有参数。
    

## 2. BERT 模型架构与输入表征

### 2.1 Transformer Encoder 架构细节

BERT 采用了 Vaswani 等人提出的经典 Transformer Encoder 结构。其核心运算为多头自注意力机制（Multi-Head Self-Attention）：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

论文中重点评估了两个不同配置的模型：

-   **$\text{BERT}_{\text{BASE}}$**：
    
    -   层数 ($L$) = 12
        
    -   隐层维度 ($H$) = 768
        
    -   自注意力头数 ($A$) = 12
        
    -   总参数量 = 110M（与 OpenAI GPT 参数量保持一致，便于对比）
        
-   **$\text{BERT}_{\text{LARGE}}$**：
    
    -   层数 ($L$) = 24
        
    -   隐层维度 ($H$) = 1024
        
    -   自注意力头数 ($A$) = 16
        
    -   总参数量 = 340M
        

### 2.2 统一输入表示 (Input Representation)

为了同时灵活支持单句和句对（Sentence Pair）任务，BERT 的输入序列由三个 Embedding 的逐元素相加（Element-wise Addition）构成：

$$\mathbf{E}_{\text{Input}} = \mathbf{E}_{\text{Token}} + \mathbf{E}_{\text{Segment}} + \mathbf{E}_{\text{Position}}$$

```
Token Embeddings:    [CLS]   my     dog    is    cute   [SEP]    he    likes   play   ##ing  [SEP]
                      +      +      +      +      +       +      +      +      +      +      +
Segment Embeddings:   EA     EA     EA     EA     EA      EA     EB     EB     EB     EB     EB
                      +      +      +      +      +       +      +      +      +      +      +
Position Embeddings:  E0     E1     E2     E3     E4      E5     E6     E7     E8     E9     E10

```

-   **Token Embeddings**：采用 WordPiece 分词（词表大小 30,000）。遇到未登录词或衍生词时拆分为子词（例如 `playing` $\rightarrow$ `play` + `##ing`）。
    
-   **Segment Embeddings**：用于区分句子 A 和句子 B。对于单句任务，全部使用 $E_A$。
    
-   **Position Embeddings**：可学习的位置向量（与原始 Transformer 使用的固定正弦/余弦位置编码不同），最大支持序列长度为 512。
    

### 2.3 特殊 Token 的作用

-   **`[CLS]` Token**：置于整个输入序列的第一个位置。经过 Transformer 的深层交互后，该位置的输出特征向量 $C \in \mathbb{R}^H$ 被用作整个序列的综合语义表示（Aggregate Representation），常用于句子分类任务。
    
-   **`[SEP]` Token**：作为特殊分隔符，用于在文本对中分隔句子 A 与句子 B，或者作为序列结尾的标记。
    

## 3. 预训练任务深度拆解

为了在不打破“双向上下文”的前提下训练模型，BERT 设计了两个联合自监督预训练任务。

### 3.1 掩码语言模型（Masked Language Model, MLM）

#### 3.1.1 掩码机制与概率分配

传统的自回归语言模型无法直接应用深层双向自注意力，因为这会导致目标词“看见”自己。BERT 借鉴了 Cloze（完形填空）测试，在每条训练数据中随机选择 **15%** 的 WordPiece Token 进行掩码处理。

为了缩小预训练阶段（包含 `[MASK]`）与下游微调阶段（无 `[MASK]`）的分布差异（Train-Test Gap），这 15% 被选中的 Token 会遵循以下细分替换规则：

-   **80% 的概率**：替换为固定的 `[MASK]` 标记。
    
    -   _例_：`my dog is cute` $\rightarrow$ `my dog is [MASK]`
        
-   **10% 的概率**：替换为词表中的任意随机 Token（引入噪声，增强模型的抗噪鲁棒性）。
    
    -   _例_：`my dog is cute` $\rightarrow$ `my dog is apple`
        
-   **10% 的概率**：保持原 Token 不变（逼迫模型时刻保持对真实输入词的表征能力）。
    
    -   _例_：`my dog is cute` $\rightarrow$ `my dog is cute`
        

#### 3.1.2 损失计算原理

对于被 Mask 的词，模型通过顶层 Transformer 输出的隐藏状态向量 $\mathbf{T}_i \in \mathbb{R}^H$，连接一个带有 Softmax 的分类器预测原始 Token：

$$P(w_i \vert{} \mathbf{X}) = \text{Softmax}(\mathbf{W}_{\text{vocab}} \mathbf{T}_i + \mathbf{b}_{\text{vocab}})$$

$\mathcal{L}_{\text{MLM}}$ 即为所有被选中的 Mask 位置上的交叉熵损失平均值。

### 3.2 下句预测任务（Next Sentence Prediction, NSP）

许多下游任务（如问答 SQuAD、自然语言推理 MNLI）依赖于理解**两个句子之间的逻辑关系**，这是单句语言模型难以捕获的。

#### 3.2.1 采样机制与标签

-   在构建预训练样本对 $(A, B)$ 时：
    
    -   **50% 的样本**：句子 $B$ 是语料库中紧随句子 $A$ 的真实下一句，标签为 `IsNext`。
        
    -   **50% 的样本**：句子 $B$ 是从语料库中随机采样的无关句子，标签为 `NotNext`。
        

#### 3.2.2 分类计算

利用 `[CLS]` 位置的隐状态向量 $C \in \mathbb{R}^H$，乘以分类权重矩阵 $\mathbf{W}_{\text{NSP}} \in \mathbb{R}^{2 \times H}$：

$$P(\text{IsNext} \vert{} A, B) = \text{Softmax}(\mathbf{W}_{\text{NSP}} C)$$

### 3.3 联合损失函数计算

预训练过程的总损失为 MLM 损失与 NSP 损失的直接相加：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$$

#### 预训练关键超参数配置：

-   **预训练数据集**：BooksCorpus (800M words) + English Wikipedia (2,500M words)。
    
-   **优化器**：Adam ($\beta_1 = 0.9, \beta_2 = 0.999$), L2 Weight Decay = 0.01。
    
-   **学习率策略**：前 10,000 steps 进行 Warmup，最高学习率 $1e-4$，之后线性衰减。
    
-   **激活函数**：GELU (Gaussian Error Linear Unit)。
    
-   **Batch Size**：256 个序列，序列长度为 512（即每 Batch 包含约 $128,000$ 个 Token），训练 1,000,000 steps（约 40 个 Epoch）。
    

## 4. 下游任务微调范式（Fine-Tuning Strategies）

微调是 BERT 的核心亮点：只需替换顶层输出层，并将所有预训练参数结合下游任务数据一起进行端到端（End-to-End）更新。

```
    (a) 句对分类任务 (如 MNLI)                (b) 单句分类任务 (如 SST-2)
          [Class Label]                            [Class Label]
               ^                                         ^
            [ [CLS] ]                                 [ [CLS] ]
         +--------------+                          +--------------+
         |     BERT     |                          |     BERT     |
         +--------------+                          +--------------+
          Tok1 ... TokN                             Tok1 ... TokN

    (c) 抽取式问答 (如 SQuAD)                 (d) 序列标注任务 (如 NER)
          [Start/End Span]                         [Label1] [Label2] ...
               ^                                       ^        ^
            [ T1 ... TN ]                            [ T1  ...  TN ]
         +--------------+                          +--------------+
         |     BERT     |                          |     BERT     |
         +--------------+                          +--------------+
          Question Paragraph                         Tok1  ...  TokN

```

### 4.1 句对分类任务

-   **代表任务**：MNLI、QQP、RTE、SWAG。
    
-   **输入构建**：`[CLS] Sentence A [SEP] Sentence B [SEP]`。
    
-   **逻辑实现**：取 `[CLS]` 隐藏状态 $C \in \mathbb{R}^H$，过分类层 $W \in \mathbb{R}^{K \times H}$（$K$ 为类别数）：
    

$$\mathbf{y} = \text{Softmax}(W C^T)$$

### 4.2 单句分类任务

-   **代表任务**：SST-2（情感分析）、CoLA（语法判定）。
    
-   **输入构建**：`[CLS] Sentence [SEP]`。
    
-   **逻辑实现**：与句对分类完全一致，仅输入中只有单个句子。
    

### 4.3 抽取式问答任务 (Question Answering)

-   **代表任务**：SQuAD v1.1 / v2.0。
    
-   **输入构建**：`[CLS] Question [SEP] Paragraph [SEP]`。
    
-   **逻辑实现**：需要预测答案在 Paragraph 中的起始位置 $S$ 与结束位置 $E$。
    
    -   引入两个可学习的向量：起始向量 $S \in \mathbb{R}^H$ 与结束向量 $E \in \mathbb{R}^H$。
        
    -   段落中第 $i$ 个 Token 作为起始位置的概率为：
        

$$P_i = \frac{e^{S \cdot \mathbf{T}_i}}{\sum_j e^{S \cdot \mathbf{T}_j}}$$

-   对于 SQuAD v2.0（允许无解），将无解情况处理为起始与结束位置均落在 `[CLS]` Token 上。
    

### 4.4 序列标注任务

-   **代表任务**：CoNLL-2003 NER（命名实体识别）。
    
-   **输入构建**：`[CLS] Sequence [SEP]`。
    
-   **逻辑实现**：将每个输入 Token 对应位置的输出向量 $\mathbf{T}_i \in \mathbb{R}^H$ 送入 Token 级别的 Softmax 分类器，预测相应的标签（如 B-PER、I-PER、O 等）。
    

## 5. 基于 Feature-based 的提取模式与消融实验

### 5.1 Fine-Tuning vs. Feature-Based (以 CoNLL-2003 NER 为例)

虽然微调模式是 BERT 的首选，但在某些特定场景下（如无法更改模型架构或需要预先计算特征以节省计算资源），特征提取模式（Feature-based Approach）同样高效。

在 CoNLL-2003 NER 任务上的对比分析：

-   **Fine-Tuning 模式 ($\text{BERT}_{\text{LARGE}}$)**：获得 **92.8%** 的 Test F1 评分。
    
-   **Feature-Based 模式 ($\text{BERT}_{\text{BASE}}$ 提取向量送入 2 层 BiLSTM)**：
    
    -   仅使用最顶层 hidden state：F1 = 94.9% (Dev)
        
    -   拼接最后 4 层的隐藏状态 (Concat Last Four Hidden Layers)：获得 **96.1%** 的 Dev F1（Test F1 与全量微调相差仅 0.3%）。
        
-   **结论**：BERT 作为通用特征提取器（类似 ELMo）同样非常出色。
    

### 5.2 模型规模与消融分析 (Ablation Studies)

根据 BERT 论文中的消融实验数据，总结出以下关键结论：

-   **模型规模效应**：
    
    -   随着层数 $L$ 和隐藏维度 $H$ 的增加，下游任务指标持续提升。
        
    -   即使在极小数据集（如 MRPC）上，超大模型（$\text{BERT}_{\text{LARGE}}$）也能通过预训练参数实现显著收益，未出现严重的过拟合。
        
-   **NSP 任务的有效性**：
    
    -   去除 NSP 任务后，在 NLI（如 MNLI）和问答（SQuAD）等需句间逻辑的任务上性能明显下降。
        
    -   注：在 BERT 之后的研究（如 RoBERTa）中，通过改变采样方式证明了去除 NSP 也能达到相当性能，但在 BERT 原始设计中，NSP 提供了关键的跨句上下文抽象能力。
        

## 6. 总结与后续演进

### 6.1 BERT 的三大历史贡献

1.  **打破单向范式**：通过双向 Transformer Encoder 与 Masked LM，首次证明了深层联合双向表征相比单向语言模型的巨大优势。
    
2.  **极大统一了 NLP 任务范式**：摒弃了针对不同下游任务单独设计复杂模型（如 Attention 机制、BiLSTM 堆叠）的传统做法，实现了真正的通用预训练与极简微调。
    
3.  **推动了预训练时代的到来**：为后续大语言模型（LLM）的发展奠定了坚实的技术基础。
    

### 6.2 代码示例：计算 MLM 与 NSP 损失 (PyTorch 逻辑实现)

Python

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class BERTLoss(nn.Module):
    def __init__(self, vocab_size):
        super(BERTLoss, self).__init__()
        # MLM 交叉熵损失，忽略未被 Mask 的位置 (-100)
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        # NSP 二分类交叉熵损失
        self.nsp_loss_fn = nn.CrossEntropyLoss()

    def forward(self, mlm_logits, nsp_logits, mlm_labels, nsp_labels):
        """
        mlm_logits: [Batch_Size, Seq_Len, Vocab_Size]
        nsp_logits: [Batch_Size, 2]
        mlm_labels: [Batch_Size, Seq_Len] - 被 mask 位置为原 token id，其余位置为 -100
        nsp_labels: [Batch_Size] - 1 代表 IsNext, 0 代表 NotNext
        """
        # 1. 计算 MLM Loss
        mlm_loss = self.mlm_loss_fn(
            mlm_logits.view(-1, mlm_logits.size(-1)), 
            mlm_labels.view(-1)
        )
        
        # 2. 计算 NSP Loss
        nsp_loss = self.nsp_loss_fn(
            nsp_logits, 
            nsp_labels
        )
        
        # 3. 总损失直接相加
        total_loss = mlm_loss + nsp_loss
        return total_loss, mlm_loss, nsp_loss
```