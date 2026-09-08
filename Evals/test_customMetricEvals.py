# Assertion - Judge LLM (Expected output) -> Actual output
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams
from deepeval.tracing import observe, update_current_trace
from agent_instrumented import support_agent as _support_agent
from judge import get_judge


@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)
    return _support_agent(user_input)


correctness = GEval(
    name="Correctness",
    criteria=(
        "Determine whether the actual output conveys the same factual information "
        "as the expected output. Minor wording differences are acceptable; "
        "missing or wrong facts are not."
    ),
    model=get_judge(),
    threshold=0.79,
    async_mode=False,
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],
)

dataset = EvaluationDataset(goldens=[
    Golden(
        input="Where is my order ORD-1042?",
        expected_output="order ORD-1042 is shipped and will arrive by May 13th",
    )
])


def test_custom_metric_correctness():
    """GEval Correctness against expected_output."""
    async_config = AsyncConfig(run_async=False, max_concurrent=1, throttle_value=2.0)
    for golden in dataset.evals_iterator(
        metrics=[correctness],
        async_config=async_config,
    ):
        support_agent(golden.input)
        time.sleep(1)


if __name__ == "__main__":
    test_custom_metric_correctness()
