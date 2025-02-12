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
使用Distilabel生成数据的模块。
Module for generating data using Distilabel.
"""

from typing import Optional

from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration


def build_distilabel_pipeline(
    model: str,
    base_url: str = "http://localhost:8000/v1",
    prompt_column: Optional[str] = None,
    prompt_template: str = "{{ instruction }}",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: int = 8192,
    num_generations: int = 1,
    input_batch_size: int = 64,
    client_replicas: int = 1,
    timeout: int = 900,
    retries: int = 0,
) -> Pipeline:
    """
    构建Distilabel数据生成流水线。
    Build a Distilabel pipeline for data generation.

    Args:
        model: 用于生成的模型名称
        base_url: vLLM服务器的URL
        prompt_column: 提示文本所在的列名
        prompt_template: 提示模板字符串
        temperature: 生成的温度参数
        top_p: 生成的top-p参数
        max_new_tokens: 最大生成的新token数量
        num_generations: 每个问题生成的答案数量
        input_batch_size: 输入处理的批次大小
        client_replicas: 并行处理的客户端副本数
        timeout: 请求超时时间（秒）
        retries: 失败请求的重试次数

    Returns:
        Pipeline: 配置好的Distilabel流水线
    """
    # 设置生成参数
    generation_kwargs = {"max_new_tokens": max_new_tokens}

    if temperature is not None:
        generation_kwargs["temperature"] = temperature

    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    # 创建并配置流水线
    with Pipeline().ray() as pipeline:
        TextGeneration(
            llm=OpenAILLM(
                base_url=base_url,
                api_key="something",  # vLLM不需要实际的API密钥
                model=model,
                timeout=timeout,
                max_retries=retries,
                generation_kwargs=generation_kwargs,
            ),
            template=prompt_template,
            input_mappings={"instruction": prompt_column} if prompt_column is not None else {},
            input_batch_size=input_batch_size,
            num_generations=num_generations,
            group_generations=True,
            resources=StepResources(replicas=client_replicas),
        )

    return pipeline


if __name__ == "__main__":
    import argparse

    from datasets import load_dataset

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="使用DeepSeek R1运行distilabel流水线生成响应")
    parser.add_argument(
        "--hf-dataset",
        type=str,
        required=True,
        help="要加载的HuggingFace数据集",
    )
    parser.add_argument(
        "--hf-dataset-config",
        type=str,
        required=False,
        help="要使用的数据集配置",
    )
    parser.add_argument(
        "--hf-dataset-split",
        type=str,
        default="train",
        help="要使用的数据集分割",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
        help="提示文本所在的列名",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="{{ instruction }}",
        help="格式化提示的模板字符串",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="用于生成的模型名称",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM服务器的URL",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="生成的温度参数",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="生成的top-p参数",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="最大生成的新token数量",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="每个问题生成的答案数量",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=64,
        help="输入处理的批次大小",
    )
    parser.add_argument(
        "--client-replicas",
        type=int,
        default=1,
        help="并行处理的客户端副本数",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="请求超时时间（秒）（默认：600）",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="失败请求的重试次数（默认：0）",
    )
    parser.add_argument(
        "--hf-output-dataset",
        type=str,
        required=False,
        help="要推送结果的HuggingFace仓库",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="推送到HF Hub时是否将输出数据集设为私有",
    )

    args = parser.parse_args()

    # 打印运行参数
    print("\n使用以下参数运行：")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    # 加载数据集
    print(f"正在加载'{args.hf_dataset}'（配置：{args.hf_dataset_config}，分割：{args.hf_dataset_split}）数据集...")
    dataset = load_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split)
    print("数据集加载完成！")

    # 构建流水线
    pipeline = build_distilabel_pipeline(
        model=args.model,
        base_url=args.vllm_server_url,
        prompt_template=args.prompt_template,
        prompt_column=args.prompt_column,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_generations=args.num_generations,
        input_batch_size=args.input_batch_size,
        client_replicas=args.client_replicas,
        timeout=args.timeout,
        retries=args.retries,
    )

    # 运行生成流水线
    print("正在运行生成流水线...")
    distiset = pipeline.run(
        dataset=dataset,
        dataset_batch_size=args.input_batch_size * 1000,
        use_cache=False,
    )
    print("生成流水线完成！")

    # 推送结果到HuggingFace Hub（如果指定）
    if args.hf_output_dataset:
        print(f"正在将生成的数据集推送到'{args.hf_output_dataset}'...")
        distiset.push_to_hub(args.hf_output_dataset, private=args.private)
        print("数据集推送完成！")
