from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed
)
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset
import os
import json
import sys

# 导入数据集处理器
from okwinds_dataset_processor import process_okwinds_dataset, load_okwinds_dataset

# 设置随机种子确保可复现性
set_seed(42)

# 1. 加载模型和分词器
def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-3B-Instruct"):
    """加载并配置Qwen模型和分词器"""
    print(f"正在加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 使用BF16精度
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
    print("模型加载完成")
    return model, tokenizer

# 2. 准备DPO数据集
def prepare_dataset(data_path=None, split=True, train_ratio=0.9):
    """
    准备DPO训练数据集
    
    Args:
        data_path: 数据集文件路径，如果为None则使用处理器的默认行为
        split: 是否分割训练集和验证集
        train_ratio: 训练集比例
    
    Returns:
        训练集和验证集的Dataset对象
    """
    try:
        # 使用okwinds_dataset_processor处理数据集
        print("开始处理数据集...")
        processed_data = process_okwinds_dataset(
            data_path=data_path,
            split=split,
            train_ratio=train_ratio
        )
        
        # 转换为Dataset对象
        train_dataset = Dataset.from_dict({
            "prompt": [d["prompt"] for d in processed_data["train"]],
            "chosen": [d["chosen"] for d in processed_data["train"]],
            "rejected": [d["rejected"] for d in processed_data["train"]]
        })
        
        # 如果有验证集，也转换为Dataset对象
        validation_dataset = None
        if "validation" in processed_data and len(processed_data["validation"]) > 0:
            validation_dataset = Dataset.from_dict({
                "prompt": [d["prompt"] for d in processed_data["validation"]],
                "chosen": [d["chosen"] for d in processed_data["validation"]],
                "rejected": [d["rejected"] for d in processed_data["validation"]]
            })
        
        print(f"数据集准备完成: 训练集 {len(train_dataset)} 样本")
        if validation_dataset:
            print(f"验证集 {len(validation_dataset)} 样本")
        
        return train_dataset, validation_dataset
        
    except Exception as e:
        print(f"数据集准备失败: {e}")
        sys.exit(1)

# 3. 数据预处理函数
def format_dpo_examples(examples, tokenizer):
    """格式化DPO训练样本，使用tokenizer.apply_chat_template方法
    
    在DPO训练中：
    1. prompt部分需要按照模型的对话格式进行正确格式化
    2. chosen和rejected部分不需要额外的格式化，它们作为模型的生成内容直接附加到prompt后
    """
    formatted = {"prompt": [], "chosen": [], "rejected": []}
    
    for i in range(len(examples["prompt"])):
        try:
            # 获取原始文本内容
            prompt_text = examples['prompt'][i]
            chosen_text = examples['chosen'][i]  # 这是模型应该生成的优质回复
            rejected_text = examples['rejected'][i]  # 这是模型不应该生成的较差回复
            
            # ========== Prompt格式化 ==========
            # 对于prompt部分，我们需要使用apply_chat_template进行正确的对话格式处理
            # 这样模型才能正确理解用户的输入上下文
            messages = [{"role": "user", "content": prompt_text}]
            
            # 使用tokenizer的apply_chat_template方法格式化prompt
            # add_generation_prompt=True会自动添加assistant的提示标记
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # 添加"assistant"提示标记
            )
            
            # ========== DPO训练格式构建 ==========
            # 存储格式化后的prompt
            formatted["prompt"].append(formatted_prompt)
            
            # chosen和rejected回复不需要单独使用apply_chat_template处理
            # 因为在DPO训练中，它们被视为模型的直接输出内容
            # 我们只需将它们附加到已格式化的prompt后面，并添加结束标记即可
            formatted["chosen"].append(formatted_prompt + chosen_text + tokenizer.eos_token)
            formatted["rejected"].append(formatted_prompt + rejected_text + tokenizer.eos_token)
            
        except Exception as e:
            print(f"格式化样本 {i} 时出错: {e}")
            # 添加空字符串作为占位符，确保数据结构完整性
            formatted["prompt"].append("")
            formatted["chosen"].append("")
            formatted["rejected"].append("")
    
    return formatted

# 4. 配置LoRA参数
def get_lora_config():
    """获取LoRA配置"""
    lora_config = LoraConfig(
        r=8,  # LoRA注意力维度
        lora_alpha=32,  # 缩放参数
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    return lora_config

def main():
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    
    # 准备数据集
    train_dataset, validation_dataset = prepare_dataset()
    
    # 格式化数据集
    print("格式化训练集...")
    train_dataset = train_dataset.map(
        lambda examples: format_dpo_examples(examples, tokenizer),
        batched=True
    )
    
    if validation_dataset:
        print("格式化验证集...")
        validation_dataset = validation_dataset.map(
            lambda examples: format_dpo_examples(examples, tokenizer),
            batched=True
        )
    
    # 获取LoRA配置并应用到模型
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数比例
    
    # 配置训练参数
    training_args = TrainingArguments(
        per_device_train_batch_size=4,  # 不使用量化可增大batch size
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=3,
        output_dir="./dpo_lora_results",
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        evaluation_strategy="epoch" if validation_dataset else "no",
        eval_steps=200 if validation_dataset else None,
        logging_dir="./logs",
        report_to="tensorboard",
        bf16=True,  # 使用BF16精度
        gradient_checkpointing=True,  # 梯度检查点减少显存
        optim="adamw_torch",
        remove_unused_columns=False
    )
    
    # 初始化DPOTrainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # 自动创建参考模型副本
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        beta=0.1,  # DPO温度参数
        max_prompt_length=256,
        max_length=512
    )
    
    # 开始训练
    print("开始训练...")
    dpo_trainer.train()
    
    # 保存适配器权重
    output_dir = "./qwen_dpo_lora_adapter"
    dpo_trainer.save_model(output_dir)
    
    # 保存完整模型（可选）
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"训练完成！LoRA适配器已保存至: {output_dir}")

if __name__ == "__main__":
    main()