"""
GRPO训练的奖励函数模块。
Reward functions for GRPO training.
"""

import math
import re

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(completions, solution, **kwargs):
    """
    检查模型输出是否与标准答案相同的奖励函数。
    Reward function that checks if the completion is the same as the ground truth.
    
    Args:
        completions: 模型生成的完成序列列表
        solution: 标准答案列表
        **kwargs: 其他参数
    
    Returns:
        rewards: 奖励值列表，正确为1.0，错误为0.0
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # 要求答案必须使用正确的latex格式（不允许有格式错误的运算符）
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # 确保优先尝试boxed内容
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # 如果内容与标准答案相同则奖励为1，否则为0
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            # 如果标准答案无法解析，我们给予1分以跳过这个样例
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """
    检查模型输出是否符合特定格式的奖励函数。
    Reward function that checks if the completion has a specific format.
    
    格式要求：必须包含<think>...</think>和<answer>...</answer>标签
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def reasoning_steps_reward(completions, **kwargs):
    r"""
    检查是否有清晰的逐步推理过程的奖励函数。
    Reward function that checks for clear step-by-step reasoning.
    
    正则表达式模式说明：
    Regex pattern:
        Step \d+: - 匹配"Step 1:", "Step 2:"等
        ^\d+\. - 匹配行首的数字列表，如"1.", "2."等
        \n- - 匹配破折号项目符号
        \n\* - 匹配星号项目符号
        First,|Second,|Next,|Finally, - 匹配过渡词
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # 使用魔法数字3来鼓励至少3个步骤，否则给予部分奖励
    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """
    获取基于余弦缩放的奖励函数。
    Get a reward function that scales based on cosine schedule.
    
    Args:
        min_value_wrong: 错误答案的最小奖励值
        max_value_wrong: 错误答案的最大奖励值
        min_value_correct: 正确答案的最小奖励值
        max_value_correct: 正确答案的最大奖励值
        max_len: 缩放的最大长度
    """
    def cosine_scaled_reward(completions, solution, **kwargs):
        """
        基于完成序列长度使用余弦调度进行缩放的奖励函数。
        Reward function that scales based on completion length using a cosine schedule.

        较短的正确解答获得更高的奖励。
        较长的错误解答受到较少的惩罚。
        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: 模型生成的完成序列列表 List of model completions
            solution: 标准答案列表 List of ground truth solutions

        该函数由以下参数控制 This function is parameterized by the following arguments:
            min_value_wrong: 错误答案的最小奖励值 Minimum reward for wrong answers
            max_value_wrong: 错误答案的最大奖励值 Maximum reward for wrong answers
            min_value_correct: 正确答案的最小奖励值 Minimum reward for correct answers
            max_value_correct: 正确答案的最大奖励值 Maximum reward for correct answers
            max_len: 缩放的最大长度 Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # 跳过无法解析的样例 Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # 基于长度应用余弦缩放
            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # 对错误答案交换最小值和最大值
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    计算N-gram重复惩罚，如https://arxiv.org/abs/2502.03373论文附录C.2所述。
    参考实现来自：https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py
    
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
        ngram_size: n-gram的大小 size of the n-grams
        max_penalty: 最大（负）惩罚值 Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        """生成文本的n-gram序列"""
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        惩罚重复内容的奖励函数
        reward function the penalizes repetitions
        参考实现：https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: 模型生成的完成序列列表 List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            # 计算重复率并应用惩罚
            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward
