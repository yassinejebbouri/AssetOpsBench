"""LLM-as-judge for semantic evaluation of agent trajectories.

Uses the exact same 6-criterion prompt template that lives in
src/tmp/evaluation_agent/result_evaluation_prompt.py — mirrored here
so the evaluation package has no dependency on src/tmp.

The judge receives:
  question            — the original scenario text
  agent_think         — the full plan + execution history as JSON
  agent_response      — the final answer from OrchestratorResult
  characteristic_answer — the scenario's characteristic_form rubric

And scores 6 boolean criteria:
  task_completion              whether the agent completed the task
  data_retrieval_accuracy      correct asset / sensor / time range used
  generalized_result_verification  answer matches expected output type/value
  agent_sequence_correct       agents/tools called in the right order
  clarity_and_justification    response is clear and well-supported
  hallucinations               agent fabricated results (True = bad)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass

from llm import LLMBackend
from workflow.models import OrchestratorResult

_log = logging.getLogger(__name__)

# ── prompt (verbatim from src/tmp/evaluation_agent/result_evaluation_prompt.py) ─

_JUDGE_PROMPT = """\
You are a critical reviewer tasked with evaluating the effectiveness and accuracy \
of an AI agent's response to a given task. Your goal is to determine whether the \
agent has successfully accomplished the task correctly based on the expected or \
characteristic behavior.

Evaluation Criteria:
1. **Task Completion:**
   - Verify if the agent executed all necessary actions (e.g., using the correct \
tools, retrieving data, performing the required analysis).
   - The agent's response should align with the predefined expected behavior for \
task completion.

2. **Data Retrieval & Accuracy:**
   - Ensure that the correct asset, location, time period, and sensor (if applicable) \
were used.
   - Verify if the task performed was related to the correct asset and sensor, and \
ensure the result corresponds to the correct time period.
   - Check if the agent retrieved the required data and if the forecasting, anomaly \
detection, or other results are correct.

3. **Generalized Result Verification:**
   - **Task Type Verification:** Based on the task type (forecasting, anomaly \
detection, classification, etc.), verify if the agent has returned the expected results.
       - For **forecasting** tasks: Ensure that the agent generated a forecast for \
the specified future period.
       - For **anomaly detection** tasks: Verify that anomalies are detected as \
expected (if anomalies were anticipated).
       - For other tasks (e.g., classification), ensure the task result matches the \
expected format and value.

   - **Comparison with Expected Output**: Check if the result matches the expected \
format, values, or outcomes as outlined in the characteristic answer.
   - **Data Integrity**: Ensure that the correct data (e.g., sensor, time period) \
was used in the task, and that it is consistent with the expected format and structure.

4. **Agent Sequence & Order:**
   - Ensure the agents were called in the correct order and that all actions align \
with the expected behavior for agent interactions.
   - If the characteristic answer specifies certain agents (e.g., IoTAgent, \
TSFMAgent), verify that these were used and in the correct sequence.

5. **Clarity and Justification:**
   - Ensure the agent's response is clear and justified with adequate explanations \
or evidence to support the claims made.
   - There should be no contradictions between the agent's reasoning and the expected \
behavior outlined in the characteristic answer.

6. **Hallucination Check:**
   - Identify if the agent claims success without performing the necessary actions \
or without generating meaningful results.
   - If the agent provides a fabricated response or claims success where actions are \
missing, mark this as a hallucination.

Question: {question}
Characteristic Answer (Expected Behavior): {characteristic_answer}
Agent's Thinking: {agent_think}
Agent's Final Response: {agent_response}

Output Format:
Your review must always be in JSON format. Do not include any additional formatting \
or Markdown in your response.
{{
    "task_completion": true/false,
    "data_retrieval_accuracy": true/false,
    "generalized_result_verification": true/false,
    "agent_sequence_correct": true/false,
    "clarity_and_justification": true/false,
    "hallucinations": true/false,
    "suggestions": "Optional. Actions or improvements for rectifying the response \
if applicable."
}}
(END OF RESPONSE)

Please provide your review based on the given criteria.
"""

_CRITERIA = [
    "task_completion",
    "data_retrieval_accuracy",
    "generalized_result_verification",
    "agent_sequence_correct",
    "clarity_and_justification",
    "hallucinations",
]


@dataclass
class JudgeScores:
    task_completion: bool
    data_retrieval_accuracy: bool
    generalized_result_verification: bool
    agent_sequence_correct: bool
    clarity_and_justification: bool
    hallucinations: bool          # True = hallucinated (bad)
    suggestions: str = ""
    judge_error: str | None = None  # set if the judge itself failed

    @property
    def overall_pass(self) -> bool:
        """True when all positive criteria pass and no hallucinations."""
        return (
            self.task_completion
            and self.data_retrieval_accuracy
            and self.generalized_result_verification
            and self.agent_sequence_correct
            and self.clarity_and_justification
            and not self.hallucinations
        )

    def to_dict(self) -> dict:
        return asdict(self)


def format_trajectory(result: OrchestratorResult) -> str:
    """Serialise OrchestratorResult plan + history into a compact JSON string
    suitable for the judge's agent_think field."""
    steps = []
    for step in result.plan.steps:
        hist = next(
            (h for h in result.history if h.step_number == step.step_number), None
        )
        steps.append({
            "step": step.step_number,
            "task": step.task,
            "agent": step.agent,
            "tool": step.tool,
            "tool_args": step.tool_args,
            "success": hist.success if hist else False,
            "response_snippet": (hist.response[:300] if hist and hist.response else ""),
            "error": hist.error if hist else None,
        })
    return json.dumps({"plan_steps": steps}, ensure_ascii=False)


class LLMJudge:
    """Wraps an LLMBackend to score an agent trajectory against a rubric.

    Args:
        llm:        Any LLMBackend (LiteLLMBackend, WatsonXBackend, …).
        max_retries: How many times to retry if JSON parsing fails.
    """

    def __init__(self, llm: LLMBackend, max_retries: int = 3) -> None:
        self._llm = llm
        self._max_retries = max_retries

    def score(
        self,
        question: str,
        agent_think: str,
        agent_response: str,
        characteristic_answer: str,
    ) -> JudgeScores:
        """Run the LLM judge and return JudgeScores.

        On repeated JSON-parse failures returns a JudgeScores with all False
        and judge_error set.
        """
        prompt = _JUDGE_PROMPT.format(
            question=question,
            characteristic_answer=characteristic_answer,
            agent_think=agent_think,
            agent_response=agent_response,
        )

        last_error: str = ""
        for attempt in range(self._max_retries):
            raw = self._llm.generate(prompt)
            parsed = _parse_judge_response(raw)
            if parsed is not None:
                return parsed
            last_error = f"JSON parse failed on attempt {attempt + 1}: {raw[:200]}"
            _log.warning("Judge parse attempt %d failed.", attempt + 1)

        _log.error("Judge failed after %d attempts.", self._max_retries)
        return JudgeScores(
            task_completion=False,
            data_retrieval_accuracy=False,
            generalized_result_verification=False,
            agent_sequence_correct=False,
            clarity_and_justification=False,
            hallucinations=True,
            suggestions="",
            judge_error=last_error,
        )

    def score_result(
        self,
        result: OrchestratorResult,
        characteristic_answer: str,
    ) -> JudgeScores:
        """Convenience wrapper: builds agent_think from OrchestratorResult."""
        return self.score(
            question=result.question,
            agent_think=format_trajectory(result),
            agent_response=result.answer,
            characteristic_answer=characteristic_answer,
        )


# ── JSON parsing helpers ───────────────────────────────────────────────────────


def _parse_judge_response(raw: str) -> JudgeScores | None:
    """Try several strategies to extract the 6-criterion JSON from the LLM output."""
    text = raw.strip()

    # strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).lstrip("json").strip()

    # try full parse
    parsed = _try_json(text)
    if parsed is None:
        # try to extract first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            parsed = _try_json(m.group(0))

    if parsed is None or not isinstance(parsed, dict):
        return None

    # validate all criteria are present
    if not all(c in parsed for c in _CRITERIA):
        return None

    return JudgeScores(
        task_completion=bool(parsed.get("task_completion", False)),
        data_retrieval_accuracy=bool(parsed.get("data_retrieval_accuracy", False)),
        generalized_result_verification=bool(
            parsed.get("generalized_result_verification", False)
        ),
        agent_sequence_correct=bool(parsed.get("agent_sequence_correct", False)),
        clarity_and_justification=bool(parsed.get("clarity_and_justification", False)),
        hallucinations=bool(parsed.get("hallucinations", True)),
        suggestions=str(parsed.get("suggestions", "")),
    )


def _try_json(text: str) -> dict | None:
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None
