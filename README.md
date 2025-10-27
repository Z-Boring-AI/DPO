# DPO和SFT微调Qwen模型
这个仓库提供了两种微调Qwen大语言模型的方法：Direct Preference Optimization (DPO)。通过这个项目，您可以轻松地对Qwen模型进行微调和优化，使其更好地适应特定场景。


## 什么是DPO?

**Direct Preference Optimization (直接偏好优化)** 是一种用于对齐大型语言模型的方法，它允许模型学习人类的偏好排序，而不需要通过传统的监督学习中的显式标签。

### DPO的核心优势
- **不需要传统标签**：DPO不依赖于显式的分类或回归标签
- **直接优化偏好**：直接优化模型以生成人类偏好的输出
- **高效对齐**：相比RLHF (Reinforcement Learning from Human Feedback)，DPO实现更简单，计算成本更低
- **偏好信息编码**：通过比较"好"和"坏"的回答来学习


## 两种微调方法的对比

### 数据集格式对比

#### DPO数据集格式
DPO训练需要的数据集包含以下三个字段：
```json
{
    "prompt": "问题或提示文本",
    "chosen": "对于该提示，人类更偏好的回答",
    "rejected": "对于该提示，人类不太偏好的回答"
}
```


#### DPO损失函数
```
loss = -E_{(x,y_w,y_l)} [log(sigmoid(β * (log P(y_w|x) - log P(y_l|x))))]
```
其中：
- \(x\) 是输入提示
- \(y_w\) 是首选回答(chosen)
- \(y_l\) 是次选回答(rejected)
- \(β\) 是控制KL散度权重的超参数
- \(P(y|x)\) 是模型在给定输入x的情况下生成回答y的概率



#### DPO训练目标
- **优化偏好概率比**：最大化首选回答与次选回答的概率比值
- **保持与参考模型的相似度**：通过β参数控制与原始模型的偏离程度
- **隐式学习人类偏好**：从对比数据中学习什么是好的回答
- **提升回答质量**：训练模型生成更符合人类期望的输出


## 项目结构

```
├── README.md                   # 项目说明文档
├── okwinds_dataset_processor.py  # Okwinds数据集处理工具
├── requirements.txt            # 项目依赖
├── train_dpo.py              # DPO训练脚本
```

### 核心文件说明

- **train_dpo.py**：DPO训练主脚本，包含模型加载、数据处理、训练执行等功能
- **okwinds_dataset_processor.py**：专门处理Okwinds DPO数据集的工具脚本
- **requirements.txt**：列出项目所需的所有Python依赖包


## 快速开始

## 数据集格式

DPO训练需要的数据集包含以下三个字段：

```json
{
    "prompt": "问题或提示文本",
    "chosen": "对于该提示，人类更偏好的回答",
    "rejected": "对于该提示，人类不太偏好的回答"
}
```

### Okwinds DPO数据集
下载地址：[HumanLLMs/Human-Like-DPO-Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset)

#### 使用Okwinds数据集

要使用Okwinds数据集进行训练，只需在调用`prepare_dataset`函数时指定`dataset_name="okwinds"`参数。我们提供了专门的数据集处理器`okwinds_dataset_processor.py`，它会自动处理数据加载、清洗和格式转换。



### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行微调脚本

#### 运行DPO训练
```bash
python train_dpo.py
```

### 3. 使用自定义数据集

修改`prepare_dataset()`函数，加载您自己的数据集：

```python
# 从JSON文件加载数据集
dataset = load_dataset('json', data_files={'train': '您的数据集.json'})
```

### 使用Okwinds数据集

- 使用Okwinds数据集：
  ```bash
  python -c "from train import main; import train; train.prepare_dataset = lambda: train.prepare_dataset('okwinds'); main()"
  ```
- 或者修改train.py中的main函数，将prepare_dataset()改为prepare_dataset('okwinds')

### Okwinds数据集处理器使用说明

我们提供了专门的`okwinds_dataset_processor.py`工具，用于处理Okwinds DPO数据集：

```bash
python okwinds_dataset_processor.py
```

该工具会自动：
1. 加载数据集
2. 清洗并验证数据
3. 分割为训练集和验证集（默认9:1比例）
4. 保存处理后的数据到`./processed_data/`目录
5. 生成数据集统计信息

## 配置说明

### 模型配置
- 默认使用`Qwen/Qwen2.5-3B`模型
- 启用4位量化以减少显存使用
- 支持自动设备映射

### 训练参数
- 学习率：5e-6
- 训练轮数：3
- 批次大小：1 (使用梯度累积)
- 梯度累积步数：4
- 优化器：paged_adamw_32bit

### Okwinds数据集配置
可以在`okwinds_dataset_processor.py`中调整以下参数：
- `train_ratio`：训练集和验证集的分割比例（默认0.9）
- `output_dir`：处理后数据的保存目录
- `data_path`：数据集文件路径


## 输出

### DPO训练输出
- 训练过程中的检查点保存在`./dpo_qwen_results`目录
- 最终微调后的模型保存在`./dpo_qwen_finetuned_distributed`或`./dpo_qwen_finetuned_single`目录（取决于是否使用分布式训练）


## 注意事项

1. 首次运行时，系统会自动下载Qwen模型，这可能需要一些时间
2. 如需微调其他规模的Qwen模型，请修改`load_model_and_tokenizer()`函数中的`model_name`参数
3. 对于较大的数据集，建议增加训练轮数以获得更好的效果
4. 可以通过调整`beta`参数来控制模型与原始模型的偏离程度


