"""Run only the fixed identity observer, without checkout or scientific imports."""

import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


class IdentityObserverTests(unittest.TestCase):
    def run_observer(self, changes=None):
        workflow = (Path(__file__).parents[1] / ".github/workflows/private-review-dispatch.yml").read_text()
        source = textwrap.dedent(workflow.split("python3 - <<'PY'\n", 1)[1].split("\n          PY", 1)[0])
        claims = {
            "iss": "https://token.actions.githubusercontent.com",
            "aud": "scientific-review-nmb",
            "repository": "oshaughnessy-junior/NuclearMatter-Backend",
            "repository_id": "1257038859",
            "repository_owner_id": "274802661",
            "environment": "private-review-dispatch-nmb",
            "workflow_ref": "oshaughnessy-junior/NuclearMatter-Backend/.github/workflows/private-review-dispatch.yml@refs/heads/main",
            "ref": "refs/heads/main",
            "event_name": "pull_request_target",
            "repository_visibility": "private",
            "runner_environment": "github-hosted",
            "sub": "repo:oshaughnessy-junior/NuclearMatter-Backend:environment:private-review-dispatch-nmb",
        }
        claims.update(changes or {})
        encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        with tempfile.TemporaryDirectory() as directory:
            response = Path(directory) / "response.json"
            summary = Path(directory) / "summary.md"
            response.write_text(json.dumps({"value": "header." + encoded + ".signature"}))
            env = dict(os.environ, IDENTITY_RESPONSE_PATH=str(response), GITHUB_STEP_SUMMARY=str(summary))
            for key, claim in [("REPOSITORY", "repository"), ("REPOSITORY_ID", "repository_id"),
                               ("OWNER_ID", "repository_owner_id"), ("ENVIRONMENT", "environment"),
                               ("WORKFLOW_REF", "workflow_ref"), ("AUDIENCE", "aud")]:
                # Expected values stay from the reviewed profile, not the changed token.
                defaults = {"repository": "oshaughnessy-junior/NuclearMatter-Backend", "repository_id": "1257038859",
                            "repository_owner_id": "274802661", "environment": "private-review-dispatch-nmb",
                            "workflow_ref": "oshaughnessy-junior/NuclearMatter-Backend/.github/workflows/private-review-dispatch.yml@refs/heads/main",
                            "aud": "scientific-review-nmb"}
                env["EXPECTED_" + key] = defaults[claim]
            result = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, timeout=10)
            return result, summary.read_text() if summary.exists() else ""

    def test_exact_profile_succeeds(self):
        result, summary = self.run_observer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"schema": "genesis.scientific-review.observed-actions-identity.v1"', summary)

    def test_public_visibility_still_rejected(self):
        result, summary = self.run_observer({"repository_visibility": "public"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('"mismatched_claims": ["repository_visibility"]', summary)
        self.assertNotIn('"public"', summary)
        self.assertNotIn("header.", summary)

    def test_mismatch_values_and_unknown_claims_not_disclosed(self):
        result, summary = self.run_observer({"aud": "private-untrusted-value", "repository_id": "wrong-id",
                                              "unreviewed_secret": "must-not-escape"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('"mismatched_claims": ["aud", "repository_id"]', summary)
        for secret in ["private-untrusted-value", "wrong-id", "must-not-escape", "unreviewed_secret"]:
            self.assertNotIn(secret, summary)


if __name__ == "__main__":
    unittest.main()
