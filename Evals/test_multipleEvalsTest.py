import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import PromptAlignmentMetric, StepEfficiencyMetric, AnswerRelevancyMetric
from deepeval.tracing import observe, update_current_trace
from agent_instrumented import support_agent as _support_agent


@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    return _support_agent(user_input)

# Direct Gemini judges. API keys are configured on GeminiModel, not metrics.
gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def gemini_model() -> GeminiModel:
    return GeminiModel(model="gemini-1.5-pro", api_key=gemini_api_key)


answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7, model=gemini_model(), async_mode=False
)
step_efficiency = StepEfficiencyMetric(
    threshold=0.5, model=gemini_model(), async_mode=False
)

prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=[
        "You are a friendly customer-support agent. "
        "Keep replies short and helpful."
    ],
    threshold=0.7,
    model=gemini_model(),
    async_mode=False,
)

dataset = EvaluationDataset(goldens=[
    Golden(input="Where is my order ORD-1042?"),
    Golden(input="What's the refund policy for electronics?"),
    Golden(input="I want to return order ORD-2099, what should I do?")
])

for golden in dataset.evals_iterator(metrics=[prompt_alignment, step_efficiency, answer_relevancy]):
    support_agent(golden.input)
