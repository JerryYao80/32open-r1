# Open-R1 项目核心功能和流程说明

## 项目概述
Open-R1是一个完全开源的DeepSeek-R1复现项目。该项目旨在构建R1流水线的缺失部分，使每个人都能复现并在其基础上进行开发。

## 核心功能模块

### 1. 模型训练模块
#### 1.1 SFT（Supervised Fine-Tuning）
- 功能：对预训练模型进行有监督微调
- 实现文件：`src/open_r1/sft.py`
- 主要特点：
  - 支持多GPU训练（DDP或DeepSpeed）
  - 支持ZeRO-2和ZeRO-3优化
  - 灵活的配置系统

#### 1.2 GRPO（Group Relative Policy Optimization）
- 功能：使用GRPO算法进行模型训练
- 实现文件：`src/open_r1/grpo.py`
- 主要特点：
  - 支持vLLM加速生成
  - 多GPU并行训练
  - 特别适用于数学推理任务

### 2. 模型评估模块
- 功能：评估模型在各种任务上的性能
- 实现文件：`src/open_r1/evaluate.py`
- 支持的评估任务：
  - AIME 2024
  - MATH-500
  - GPQA Diamond

### 3. 数据生成模块
- 功能：从现有模型生成合成数据
- 实现文件：`src/open_r1/generate.py`
- 使用工具：Distilabel

### 4. 奖励系统模块
- 功能：为强化学习提供奖励计算
- 实现文件：`src/open_r1/rewards.py`

## 核心流程

### 1. 模型训练流程
```
预训练模型 -> SFT训练 -> GRPO训练 -> 模型评估
```

### 2. 数据处理流程
```
原始数据 -> 数据预处理 -> 训练数据生成 -> 模型训练使用
```

### 3. 评估流程
```
模型加载 -> 任务选择 -> 性能评估 -> 结果分析
```

## 项目架构
```
src/open_r1/
├── sft.py          # 监督微调训练
├── grpo.py         # GRPO训练实现
├── evaluate.py     # 模型评估系统
├── generate.py     # 数据生成工具
├── rewards.py      # 奖励计算系统
├── configs.py      # 配置管理
└── utils/          # 工具函数
```

## 使用流程

### 1. 环境准备
1. 创建Python虚拟环境
2. 安装vLLM和其他依赖
3. 配置Hugging Face和Weights & Biases账号

### 2. 模型训练
1. 选择合适的训练配置
2. 准备训练数据
3. 启动训练任务
4. 监控训练过程

### 3. 模型评估
1. 选择评估任务
2. 配置评估参数
3. 运行评估
4. 分析评估结果

## 注意事项
1. CUDA版本要求12.4
2. 显存使用优化建议
3. 多GPU训练配置说明
4. 模型保存和加载注意事项 