from typing import Dict, List

from prompt2yolo.configs import Prompt


def normalize_prompt_weights(prompts: List[Dict[str, float]]) -> List[Prompt]:
    total_weight = sum(prompt.get("weight", 1.0) for prompt in prompts)
    return [
        Prompt(text=prompt["text"], weight=prompt.get("weight", 1.0) / total_weight)
        for prompt in prompts
    ]
