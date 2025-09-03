import hydra

from rllm.agents.tool_ast_agent import ToolASTAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import tool_calling_ast_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("apigen_mt", "train")
    test_dataset = DatasetRegistry.load_dataset("apigen_mt", "test")

    env_args = {"reward_fn": tool_calling_ast_reward_fn}
    agent_args = {"parser_name": "qwen", "system_prompt": ""}

    trainer = AgentTrainer(
        agent_class=ToolASTAgent,
        agent_args=agent_args,
        env_args=env_args,
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
