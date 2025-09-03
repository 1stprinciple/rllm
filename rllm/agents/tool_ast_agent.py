import copy
from typing import Any
import json
from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class ToolASTAgent(BaseAgent):
    """
    A tool agent that only parses AST tree to check correctness, following the BaseAgent interface.
    Always single turn.
    """

    def __init__(self, accumulate_thinking=True):
        """
        Initialize the MathAgent.
        """
        self._trajectory = Trajectory()
        self.messages = []
        self.accumulate_thinking = accumulate_thinking

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        if not self.trajectory.steps:
            # Initial problem presentation
            assert isinstance(observation, dict) and "prompt" in observation
            question = observation["prompt"]
            question = json.loads(question)
            assert isinstance(question, list)
            self.messages = question
        else:
            # Place Holder as it's always single turn.
            self.messages.append({"role": "user", "content": "Hi!"})

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's internal state based on the model's response.
        """
        self.messages.append({"role": "assistant", "content": response})
        new_step = Step(chat_completions=copy.deepcopy(self.chat_completions))
        self.trajectory.steps.append(new_step)

        return Action(action=response)

    def reset(self):
        """Reset agent state for new episode."""
        self._trajectory = Trajectory()
        self.messages = []

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return conversation history for model interaction."""
        # remove thinking from assistant messages if not accumulate_thinking except the last one
        messages = copy.deepcopy(self.messages)
        if not self.accumulate_thinking:
            for msg in messages[:-1]:
                if msg["role"] == "assistant":
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """Return complete interaction trajectory."""
        return self._trajectory

    def get_current_state(self) -> Step:
        """Returns the current step/state of the agent."""
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]
