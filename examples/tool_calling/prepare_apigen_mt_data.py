import json

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


def prepare_deepcoder_data(train_size: int = None, test_size: int = None):
    train_dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="train")
    test_dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="test")

    def preprocess_fn(messages, idx):
        return {"prompt": question, "ground_truth": tests, "data_source": "apigen_mt_5k", "uid": f"apigen_mt_5k_{idx}", "index": idx}

    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if test_size:
        test_dataset = test_dataset.select(range(min(test_size, len(test_dataset))))

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)


    train_dataset = DatasetRegistry.register_dataset("deepcoder", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("deepcoder", test_dataset, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_deepcoder_data()
    print(f"  - Train dataset: {len(train_dataset.get_data())} examples")
    print(f"  - Test dataset: {len(test_dataset.get_data())} examples")
    print(train_dataset.get_data()[0])
