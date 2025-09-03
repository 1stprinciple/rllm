"""
This module contains the RewardToolCallingASTFn class, which parses tool calls
from model outputs and evaluates them against ground truth using AST matching.
"""
import json
from collections import Counter

from rllm.parser.tool_parser.tool_parser_base import ToolParser
from rllm.parser import get_tool_parser

from rllm.rewards.reward_types import RewardConfig, RewardOutput

class RewardToolCallingASTFn:
    """
    Reward function for evaluating mathematical answers.

    This class implements the RewardFunction protocol to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __init__(self, config: RewardConfig, parser_name: str = "qwen") -> None:
        self.config = config
        parser_class: type[ToolParser] = get_tool_parser(parser_name=parser_name)
        self.tool_parser = parser_class()

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Calculate the reward for a math task based on the agent's action.

        Args:
            task_info: Dictionary containing problem, data_source, problem_type, and ground_truth
            action: The agent's response/solution

        Returns:
            RewardOutput: The calculated reward with correctness information
        """
        # Extract information from task_info
        model_response = action

        # Handle None or empty response
        if model_response is None or model_response == "":
            print("DEBUG: Empty or None response")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Extract solution.
        try:
            tool_calls = self.tool_parser.parse(model_response)
        except Exception as e:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        tool_calls = [tool_call.to_dict() for tool_call in tool_calls]
        # Process the ground truth(s)
        ground_truths = task_info.get("ground_truth", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        ground_truths = json.loads(ground_truths)

        if compare_tool_calls(tool_calls, ground_truths):
            return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        else:
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)


def compare_tool_calls(generated_tool_calls: list, gt_tool_calls: list) -> bool:
    if len(generated_tool_calls) != len(gt_tool_calls):
        return False

    generated_tool_calls_serialized = [json.dumps(item, sort_keys=True) for item in generated_tool_calls]
    gt_tool_calls_serialized = [json.dumps(item, sort_keys=True) for item in gt_tool_calls]

    result = Counter(generated_tool_calls_serialized) == Counter(gt_tool_calls_serialized)
    if not result:
        print("Tool calls mismatch")
        print("Generation: ", generated_tool_calls)
        print("Ground Truth: ", gt_tool_calls)
    return result

