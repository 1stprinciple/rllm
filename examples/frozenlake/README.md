# FrozenLake Agent Examples

This directory contains examples for training and running FrozenLake RL agents using the RLLM framework. The FrozenLake agent learns to navigate a slippery grid world environment to reach a goal while avoiding holes.

Our examples use the following:
* Qwen3-4B as the base model
* Randomly generated FrozenLake environments with varying sizes and slip probabilities
* GRPO for training

## Environment Overview

FrozenLake is a classic reinforcement learning environment where:
- **Objective**: Navigate from start position to goal position
- **Challenge**: The surface is slippery, so actions may not always succeed as intended
- **Termination**: Episode ends when reaching goal (reward +1) or falling into hole (reward 0)
- **Parameters**:
  - `size`: Grid size (e.g., 4x4, 8x8)
  - `p`: Probability of successful action execution (1-p probability of random action)
  - `seed`: Random seed for environment generation

## Model Hosting

### Option 1: Using vLLM

Start a vLLM server with OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 
```

### Option 2: Using SGLang

```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-4B \ 
    --dp-size 1 \
    --dtype bfloat16
# increase dp_size to enable data-parallel processing on multiple GPUs 
```

The server should be accessible at `http://localhost:30000/v1`

## Dataset Preparation

Prepare the FrozenLake datasets (randomly generated environments for training and testing):

```bash
cd examples/frozenlake
python prepare_frozenlake_data.py
```

This will:
- Generate 3,000 random FrozenLake environments for training
- Generate 100 random FrozenLake environments for testing
- Register both datasets with the RLLM DatasetRegistry
- Each environment has random size (2-10), slip probability (0.6-0.85), and seed

## Running Inference

Once your model server is running and datasets are prepared, you can run inference:

```bash
cd examples/frozenlake
python run_frozenlake_agent.py
```

### Configuration Options

You can modify the inference script parameters:

- `n_parallel_agents`: Number of parallel agents (default: 256)
- `model_name`: Model to use (default: "Qwen/Qwen3-4B")
- `base_url`: API server URL (default: "http://localhost:30000/v1")
- `max_response_length`: Maximum response length (default: 16384)
- `max_prompt_length`: Maximum prompt length (default: 4096)
- `temperature`: Sampling temperature (default: 0.6)
- `top_p`: Top-p sampling (default: 0.95)

The script will:
1. Load the FrozenLake test dataset (or generate if not exists)
2. Run parallel inference using the async agent execution engine
3. Evaluate results and compute success rates

## Training

### Basic Training

To train a FrozenLake agent:

```bash
cd examples/frozenlake
bash train_frozenlake_agent.sh
```
