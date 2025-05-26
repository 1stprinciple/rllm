import asyncio
import json
from copy import deepcopy

from transformers import AutoTokenizer

from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.data.dataset_types import TestDataset, TrainDataset
from rllm.data.utils import fetch_live_code_bench_system_prompt, load_dataset
from rllm.engine.async_agent_execution_engine import AsyncAgentExecutionEngine
from rllm.environments.code.competition_coding import CompetitionCodingEnv


def load_data(n=1, dataset_enum=None):
    dataset = load_dataset(dataset_enum)
    data = []
    for idx, example in enumerate(dataset):
        if isinstance(dataset_enum, TestDataset.Math):
            processed = process_math_fn(example, idx)
        else:
            processed = process_code_fn(example, idx)
        for i in range(n):
            data.append(deepcopy(processed))
    return data


def process_data(n, dataset):
    data = []
    for idx, example in enumerate(dataset):
        processed = process_code_fn(example, idx)
        for i in range(n):
            data.append(deepcopy(processed))
    return data


def process_math_fn(example, idx):
    question = example.pop("problem")
    instruction = "Let's think step by step, put your final answer within \\boxed{}, and write python to evaluate math expressions if needed."
    question = f"{question} {instruction}"
    answer = example.pop("answer")

    task = {
        "ground_truth": answer,
        "question": question,
        "idx": idx,
        "data_source": "math",
    }
    return task


def process_code_fn(example, idx):
    question = example.pop("problem")
    tests = example.pop("tests")

    if example.get("metadata", {}):
        assert (
            "func_name" in example["metadata"]
        ), f"Function name is not found, check if your LCB data is preprocessed correctly: {example['metadata']}"
        if isinstance(tests, dict):
            tests["metadata"] = example["metadata"]
        else:
            for test in tests:
                assert isinstance(test, dict), "Test is not a dict"
                test["metadata"] = example["metadata"]

    tests = json.dumps(tests)

    instruction = fetch_live_code_bench_system_prompt(
        prompt=question, starter_code=example.pop("starter_code")
    )

    task = {
        "ground_truth": tests,
        "question": instruction,
        "idx": idx,
        "data_source": "livecodebench",
    }
    return task


def evaluate_results(results):
    from collections import defaultdict

    # Create a map to store correct answers per problem
    problem_correct_map = defaultdict(int)
    problem_total_map = defaultdict(int)

    # Count correct answers for each problem
    for trajectory in results:
        problem = trajectory.steps[0].observation

        is_correct = 1 if trajectory.reward > 0 else 0

        problem_correct_map[problem] += is_correct
        problem_total_map[problem] += 1

    k = max(problem_total_map.values())

    # Calculate pass@1 and pass@16
    total_problems = len(problem_correct_map)
    pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
    pass_at_k = (
        sum(1 for problem, correct in problem_correct_map.items() if correct > 0)
        / total_problems
    )

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print(f"Average Pass@{k} Accuracy:", pass_at_k)


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Create the environment (no batch_size parameter)
    n_parallel_agents = 256

    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    envs = [CompetitionCodingEnv(max_turns=2) for _ in range(n_parallel_agents)]

    agents = [
        CompetitionCodingAgent(remove_thinking=False, max_tests=2)
        for i in range(n_parallel_agents)
    ]

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    engine = AsyncAgentExecutionEngine(
        agents=agents,
        envs=envs,
        rollout_engine=None,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=16384,
        max_prompt_length=4096,
        config=None,
        n_parallel_agents=n_parallel_agents,
        enable_thinking=True,
    )
    # engine.update_envs_and_agents(envs, agents)

    tasks = load_data(n=1, dataset_enum=TestDataset.Code.LIVECODEBENCH)

    results = asyncio.run(engine.execute_tasks(tasks))

    # RED   = "\033[31m"
    # RESET = "\033[0m"

    # for result in results:
    #     if result.reward == 0:
    #         for step in result.steps:
    #             print(f"{RED}step{RESET}: {[step.step]}\n")
    #             print(f"{RED}observation{RESET}: {[step.observation]}\n")
    #             print(f"{RED}thought{RESET}: {[step.thought]}\n")
    #             print(f"{RED}action{RESET}: {[step.action]}\n")
    #             print(f"{RED}reward{RESET}: {[step.reward]}\n")
    #             print(f"{RED}done{RESET}: {[step.done]}\n")
    #         print(f"{RED}--------------------------------{RESET}\n\n")

    evaluate_results(results)

    # results = [result.to_dict() for result in results]
    # with open("qwen3-4b-lcb-2turn-16k-new.json", "w") as f:
    #     json.dump(results, f, indent=4)
