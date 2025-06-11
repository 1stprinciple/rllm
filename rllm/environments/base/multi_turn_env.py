from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from rllm.environments.base.base_env import BaseEnv


class MultiTurnEnvironment(BaseEnv, ABC):
    """
    An environment for multi-turn interactions with LLMs.
    The environment provides a series of questions/prompts and evaluates responses using a custom reward function.
    The interaction terminates after reaching the maximum number of turns.
    """
    
    def __init__(self, 
                 task: Optional[Dict] = None,
                 max_turns: int = 3,
                 **kwargs):
        """
        Initialize the multi-turn environment.
        
        Args:
            task: Dictionary containing the task information, including at least a "questions" field
                  with a list of questions for each turn
            max_turns: Maximum number of turns before terminating the interaction
        """
        super().__init__()
        self.task = task
        self.max_turns = max_turns
        self.current_turn = 0
        self.done = False
        self.history = []
    
    def reset(self):        
        self.done = False
        self.current_turn = 0
        self.history = []
        
        return self.task, {}
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action: Response string from the LLM
            
        Returns:
            next_observation, reward, terminated, truncated, info
        """
        # Store the action in history
        self.history.append(action)
        
        # Calculate reward for the current turn using the abstract method
        reward, next_obs = self.get_reward_and_next_obs(self.task, action)
        
        # Increment turn counter
        self.current_turn += 1
        
        # Check if we've reached the maximum number of turns
        if self.current_turn >= self.max_turns:
            self.done = True
            return {}, reward, self.done, self.task
        
        return next_obs, reward, self.done, self.task
    
    @abstractmethod
    def get_reward_and_next_obs(self, task: Dict, action: Any) -> Tuple[float, Dict]:
        """
        Abstract method to compute the reward based on the task and action.
        
        Args:
            task: The task dictionary containing relevant information
            action: The action taken by the agent
            
        Returns:
            Tuple of (reward: float, metadata: Dict)
        """
        pass

    @staticmethod
    def from_dict(env_args: Dict) -> "MultiTurnEnvironment":
        if 'task' in env_args:
            task = env_args['task']
        else:
            task = env_args
        return MultiTurnEnvironment(
            task=task,
            max_turns=env_args.get("max_turns", 3)
        )