# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
群组相对策略优化(GRPO)训练脚本
Group Relative Policy Optimization (GRPO) training script
"""

import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig
from open_r1.rewards import (
    accuracy_reward,
    format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    reasoning_steps_reward,
)
from open_r1.utils.callbacks import get_callbacks
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    GRPO训练脚本的参数配置类
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            奖励函数列表。可选值：'accuracy'(准确率), 'format'(格式), 'reasoning_steps'(推理步骤), 
            'cosine'(余弦相似度), 'repetition_penalty'(重复惩罚)
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'.
        cosine_min_value_wrong (`float`):
            错误答案的最小余弦奖励值
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            错误答案的最大余弦奖励值
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            正确答案的最小余弦奖励值
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            正确答案的最大余弦奖励值
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            余弦缩放的最大长度
            Maximum length for cosine scaling.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "奖励函数列表。可选值：'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "错误答案的最小奖励值"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "错误答案的最大奖励值"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "正确答案的最小奖励值"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "正确答案的最大奖励值"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "缩放的最大长度"},
    )

    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "重复惩罚奖励的n-gram大小"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "重复惩罚的最大（负）惩罚值"},
    )


# 系统提示词：定义助手的行为模式
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    """
    GRPO训练的主函数
    Main function for GRPO training

    Args:
        script_args: GRPO脚本参数，包含奖励函数配置等
        training_args: 训练参数，包含学习率、批次大小等
        model_args: 模型参数，包含模型路径等
    """
    # 设置随机种子以确保可重现性
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # 设置日志记录 Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 在每个进程上记录简要信息
    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # 检查是否存在上次的检查点
    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # 加载数据集
    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # 获取奖励函数
    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,  # 准确率奖励
        "format": format_reward,      # 格式奖励
        "reasoning_steps": reasoning_steps_reward,  # 推理步骤奖励
        "cosine": get_cosine_scaled_reward(        # 余弦相似度奖励
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(  # 重复惩罚奖励
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # 格式化为对话形式
    # Format into conversation
    def make_conversation(example):
        """将示例转换为对话格式"""
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # 转换数据集格式
    dataset = dataset.map(make_conversation)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # 初始化模型参数
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # 初始化GRPO训练器 Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # 训练循环 Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # 保存模型并创建模型卡片 Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # 在主进程上保存其他内容
    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # 恢复k,v缓存以加速推理
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # 评估 Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # 推送到hub push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
