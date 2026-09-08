import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import PromptAlignmentMetric, StepEfficiencyMetric, AnswerRelevancyMetric
from deepeval.tracing import observe
from agent_instrumented import support_agent as _support_agent
from judge import get_judges


@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    return _support_agent(user_input)


# One judge model per metric → separate free-tier RPM buckets.
j1, j2, j3 = get_judges(3)
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=j1, async_mode=False)
step_efficiency = StepEfficiencyMetric(threshold=0.5, model=j2, async_mode=False)
prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=[
        "You are a friendly customer-support agent. "
        "Keep replies short and helpful."
    ],
    threshold=0.7,
    model=j3,
    async_mode=False,
)

dataset = EvaluationDataset(goldens=[
    Golden(input="Where is my order ORD-1042?"),
    Golden(input="What's the refund policy for electronics?"),
    Golden(input="I want to return order ORD-2099, what should I do?"),
])


def test_multiple_evals():
    """AnswerRelevancy + StepEfficiency + PromptAlignment."""
    async_config = AsyncConfig(
        run_async=False,
        max_concurrent=1,
        throttle_value=5.0,
    )
    for golden in dataset.evals_iterator(
        metrics=[prompt_alignment, step_efficiency, answer_relevancy],
        async_config=async_config,
    ):
        support_agent(golden.input)
        time.sleep(5)  # stay under Gemini free-tier RPM


if __name__ == "__main__":
    test_multiple_evals()
