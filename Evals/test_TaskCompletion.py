import os
import sys
from dotenv import load_dotenv

# Local evaluation runs do not need DeepEval's anonymous PostHog telemetry.
# This must be set before importing DeepEval.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

from agent_instrumented import support_agent


def test_task_completion():
    user_input = "Where is my order ORD-1042?"
    actual_output = support_agent(user_input)

    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
    )

    judge = GeminiModel(
        model="gemini-3.5-flash-lite",
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    )

    evaluate(
        test_cases=[test_case],
        metrics=[TaskCompletionMetric(threshold=0.7, model=judge)],
    )


if __name__ == "__main__":
    test_task_completion()
