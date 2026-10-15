"""Exercise the browser approval flow without external API calls."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from exercises import exercise_4_audit as exercise_4
from tests.test_workflows import FakeLLM, sample_pr


class StreamlitTests(unittest.TestCase):
    def test_approval_card_resumes_and_posts_once(self):
        url = "https://github.com/example/demo/pull/1"
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"HITL_DB_PATH": str(Path(directory) / "audit.db")}), \
                patch.object(exercise_4, "fetch_pr", side_effect=sample_pr), \
                patch.object(exercise_4, "get_llm", return_value=FakeLLM(0.65)), \
                patch.object(exercise_4, "post_review_comment", return_value="https://github.com/example/demo/pull/1#issuecomment-1") as post:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
            self.assertFalse(app.exception)
            app.text_input[0].set_value(url)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Approval requested" in x.value for x in app.subheader))
            post.assert_not_called()
            next(button for button in app.button if button.label == "Approve").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(post.call_count, 1)
            self.assertTrue(app.success)


if __name__ == "__main__":
    unittest.main()
