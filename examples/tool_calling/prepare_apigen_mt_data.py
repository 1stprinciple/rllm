import json

import datasets
from typing import Any

from rllm.data.dataset import DatasetRegistry

from rllm.parser import get_tool_parser
from rllm.parser.tool_parser.tool_parser_base import ToolParser



def remove_nulls_recursive(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: remove_nulls_recursive(v) for k, v in value.items() if v is not None}
    elif isinstance(value, list):
        return [remove_nulls_recursive(x) for x in value]
    return value

def prepare_apigen_mt_data(train_size: int = None, test_size: int = None, parser_name: str = "qwen"):
    train_dataset = datasets.load_from_disk("/Users/tianyi/dataset/data/")
    test_dataset = datasets.load_from_disk("/Users/tianyi/data/")

    parser_class: type[ToolParser] = get_tool_parser(parser_name=parser_name)
    tool_parser = parser_class()

    def preprocess_fn(example, idx):
        messages = example["messages"]
        messages = remove_nulls_recursive(messages)
        tools = example.get("tools", [])
        tools = remove_nulls_recursive(tools)
        last_turn_tool_calls = messages[-1]['tool_calls']

        ground_truth = []
        for tool_call in last_turn_tool_calls:
            tool_call = tool_call["function"]
            tool_call["arguments"] = json.loads(tool_call["arguments"])
            if isinstance(tool_call["arguments"], str):
                tool_call["arguments"] = json.loads(tool_call["arguments"])
            ground_truth.append(tool_call)

        # for tool_call in ground_truth:
        #     if isinstance(tool_call["arguments"], str):
        #         print("Something wrong")
        #     elif isinstance(tool_call["arguments"], dict):
        #         print("Something right")

        tools_prompt = tool_parser.get_tool_prompt(json.dumps(tools))

        possible_system_message = messages[0]
        if possible_system_message["role"] == "system":
            system_content = possible_system_message["content"]
            messages[0]["content"] = system_content + tools_prompt
        else:
            system_message = {"role": "system", "content": system_content}
            messages = [system_message] + messages

        return {
            "prompt": json.dumps(messages[:-1]),
            "ground_truth": json.dumps(ground_truth),
            "data_source": "apigen_mt",
        }
    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if test_size:
        test_dataset = test_dataset.select(range(min(test_size, len(test_dataset))))

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)

    train_dataset = DatasetRegistry.register_dataset("apigen_mt", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("apigen_mt", test_dataset, "test")
    return train_dataset, test_dataset

if __name__ == "__main__":
    train_dataset, test_dataset = prepare_apigen_mt_data(test_size=100)
    print(f"  - Train dataset: {len(train_dataset.get_data())} examples")
    print(f"  - Test dataset: {len(test_dataset.get_data())} examples")
    # print(train_dataset.get_data()[0])
