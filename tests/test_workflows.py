"""Offline end-to-end checks for routing, resume, and the durable audit trail."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import replay_events
from common.github import PullRequest
from common.schemas import PRAnalysis, ReviewComment
from exercises import exercise_1_confidence as exercise_1
from exercises import exercise_2_hitl as exercise_2
from exercises import exercise_3_escalation as exercise_3
from exercises import exercise_4_audit as exercise_4


def sample_pr(url: str) -> PullRequest:
    return PullRequest(
        url=url, owner="example", repo="demo", number=int(url.rsplit("/", 1)[1]),
        title="Sample change", author="author", base_ref="main", head_ref="feature",
        head_sha="abcdef123456", diff="diff --git a/app.py b/app.py\n+test change",
        files_changed=["app.py"],
    )


def analysis(confidence: float) -> PRAnalysis:
    return PRAnalysis(
        summary="Review summary", risk_factors=["Input validation"],
        comments=[ReviewComment(file="app.py", line=1, severity="issue", body="Check input")],
        confidence=confidence, confidence_reasoning="Checked the changed code",
        escalation_questions=["Why does app.py accept unvalidated input?"],
    )


class FakeLLM:
    def __init__(self, initial: float):
        self.initial = initial

    def with_structured_output(self, schema):
        assert schema is PRAnalysis
        return self

    def invoke(self, messages):
        prompt = messages[-1]["content"]
        return analysis(0.7 if "Initial analysis:" in prompt else self.initial)

    async def ainvoke(self, messages):
        return self.invoke(messages)


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    def test_exercise_1_routes_three_buckets(self):
        url = "https://github.com/example/demo/pull/1"
        for confidence, decision, final_action in [
            (0.8, "auto_approve", "auto_approved"),
            (0.65, "human_approval", "pending_human_approval"),
            (0.4, "escalate", "pending_escalation"),
        ]:
            with self.subTest(confidence=confidence), \
                    patch.object(exercise_1, "fetch_pr", side_effect=sample_pr), \
                    patch.object(exercise_1, "get_llm", return_value=FakeLLM(confidence)):
                result = exercise_1.build_graph().invoke({"pr_url": url})
                self.assertEqual(result["decision"], decision)
                self.assertEqual(result["final_action"], final_action)

    def test_exercise_2_approval_resumes_without_duplicate_post(self):
        url = "https://github.com/example/demo/pull/1"
        with patch.object(exercise_2, "fetch_pr", side_effect=sample_pr), \
                patch.object(exercise_2, "get_llm", return_value=FakeLLM(0.65)), \
                patch.object(exercise_2, "post_review_comment") as post:
            graph = exercise_2.build_graph()
            cfg = {"configurable": {"thread_id": "approval-test"}}
            paused = graph.invoke({"pr_url": url}, cfg)
            self.assertEqual(paused["__interrupt__"][0].value["kind"], "approval_request")
            post.assert_not_called()
            result = graph.invoke(Command(resume={"choice": "approve", "feedback": ""}), cfg)
            self.assertEqual(result["final_action"], "committed")
            post.assert_called_once()

    def test_exercise_3_escalation_uses_answers(self):
        url = "https://github.com/example/demo/pull/2"
        with patch.object(exercise_3, "fetch_pr", side_effect=sample_pr), \
                patch.object(exercise_3, "get_llm", return_value=FakeLLM(0.4)), \
                patch.object(exercise_3, "post_review_comment") as post:
            graph = exercise_3.build_graph()
            cfg = {"configurable": {"thread_id": "escalation-test"}}
            paused = graph.invoke({"pr_url": url}, cfg)
            question = paused["__interrupt__"][0].value["questions"][0]
            result = graph.invoke(Command(resume={question: "Legacy behavior"}), cfg)
            self.assertEqual(result["final_action"], "committed_after_escalation")
            self.assertEqual(result["analysis"].confidence, 0.7)
            self.assertIn("Legacy behavior", post.call_args.args[1])

    async def test_exercise_4_persists_resume_and_audit(self):
        url = "https://github.com/example/demo/pull/2"
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {
                    "HITL_DB_PATH": str(Path(directory) / "audit.db"),
                    "GITHUB_USER": "reviewer",
                }), \
                patch.object(exercise_4, "fetch_pr", side_effect=sample_pr), \
                patch.object(exercise_4, "get_llm", return_value=FakeLLM(0.4)), \
                patch.object(exercise_4, "post_review_comment", return_value="https://github.com/example/demo/pull/2#issuecomment-1") as post:
            cfg = {"configurable": {"thread_id": "durable-test"}}
            async with AsyncSqliteSaver.from_conn_string(os.environ["HITL_DB_PATH"]) as cp:
                await cp.setup()
                paused = await exercise_4.build_graph(cp).ainvoke(
                    {"pr_url": url, "thread_id": "durable-test"}, cfg,
                )
            question = paused["__interrupt__"][0].value["questions"][0]
            post.assert_not_called()
            async with AsyncSqliteSaver.from_conn_string(os.environ["HITL_DB_PATH"]) as cp:
                await cp.setup()
                graph = exercise_4.build_graph(cp)
                snapshot = await graph.aget_state(cfg)
                self.assertEqual(snapshot.tasks[0].interrupts[0].value["kind"], "escalation")
                result = await graph.ainvoke(
                    Command(resume={question: "Legacy behavior"}), cfg,
                )
            self.assertEqual(result["final_action"], "committed")
            self.assertEqual(result["posted_comment_url"], "https://github.com/example/demo/pull/2#issuecomment-1")
            post.assert_called_once()
            events = await replay_events("durable-test")
            self.assertEqual([e["action"] for e in events], [
                "fetch_pr", "analyze", "route", "escalate", "escalate",
                "escalate", "synthesize", "commit",
            ])
            self.assertEqual(events[5]["reviewer_id"], "reviewer")
            self.assertEqual(events[6]["confidence"], 0.7)
            self.assertEqual(events[-1]["decision"], "escalate")


if __name__ == "__main__":
    unittest.main()
