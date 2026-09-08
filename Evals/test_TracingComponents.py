import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace

from agent_instrumented import support_agent as _support_agent
from judge import get_judge


@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_tools:
            update_current_trace(expected_tools=golden.expected_tools)
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)
    return _support_agent(user_input)


judge = get_judge()
task_completion = TaskCompletionMetric(threshold=0.7, model=judge, async_mode=False)
tool_correctness = ToolCorrectnessMetric()

dataset = EvaluationDataset(goldens=[
    Golden(
        input="Where is my order ORD-1042?",
        expected_tools=[ToolCall(name="get_order_status")],
    ),
    Golden(
        input="What is refund policy for electronics?",
        expected_tools=[ToolCall(name="get_refund_policy")],
    ),
])


def test_tracing_components():
    """TaskCompletion + ToolCorrectness via traced goldens."""
    async_config = AsyncConfig(run_async=False, max_concurrent=1, throttle_value=2.0)
    for golden in dataset.evals_iterator(
        metrics=[task_completion, tool_correctness],
        async_config=async_config,
    ):
        support_agent(golden.input)
        time.sleep(2)


if __name__ == "__main__":
    test_tracing_components()
