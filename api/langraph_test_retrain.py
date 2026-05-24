"""
Testovi za Korak 4: node_human_confirm, node_trigger_retrain, _patch_hyperparam_file.

Pokretanje:
    python langraph_test_retrain.py

Testovi ne zahtevaju Ollamu ni pravi model — rade sa mock state-om.
"""

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Pomoćni mock state
# ---------------------------------------------------------------------------

def _make_state(decision: str = "SCAN_FIRST", mcc: float = 0.78, confirmed: bool = False) -> dict:
    """Kreira minimalni MetricsState za testove koji ne prolaze kroz Ollama čvorove."""
    return {
        "classification_report": {},
        "confusion_matrix":      [],
        "mcc_score":             mcc,
        "roc_auc_scores":        {},
        "class_labels":          [],
        "previous_metrics":      {},
        "metrics_delta":         {},
        "regression_detected":   False,
        "per_class_analysis":    "",
        "confusion_analysis":    "",
        "weak_classes":          ["syn_flood", "ack_flood"],
        "recommendations":       {},
        "summary":               "",
        "decision":              decision,
        "scanned_files":         {},
        "proposed_hyperparams":  {
            "hidden_size":   256,
            "num_layers":    2,
            "dropout":       0.25,
            "learning_rate": 0.001,
            "seq_len":       30,
            "_reasoning":    "Test reasoning — larger hidden size to improve weak classes.",
        },
        "human_confirmed":       confirmed,
        "retrain_triggered":     False,
        "retrain_command":       "",
    }


# ---------------------------------------------------------------------------
# Test: _validate_hyperparams
# ---------------------------------------------------------------------------

class TestValidateHyperparams(unittest.TestCase):
    """Provera da validacija drži vrednosti unutar HYPERPARAMETER_SPACE."""

    def setUp(self):
        from langraph import _validate_hyperparams
        self.validate = _validate_hyperparams

    def test_valid_choice_passes_through(self):
        result = self.validate({"hidden_size": 256, "num_layers": 2, "seq_len": 30})
        self.assertEqual(result["hidden_size"], 256)
        self.assertEqual(result["num_layers"], 2)
        self.assertEqual(result["seq_len"], 30)

    def test_invalid_choice_clamped_to_nearest(self):
        # 512 nije u [64, 128, 256] — treba biti clamped na 256
        result = self.validate({"hidden_size": 512})
        self.assertEqual(result["hidden_size"], 256)

    def test_range_value_within_bounds(self):
        result = self.validate({"dropout": 0.25, "learning_rate": 0.005})
        self.assertAlmostEqual(result["dropout"], 0.25)
        self.assertAlmostEqual(result["learning_rate"], 0.005)

    def test_range_value_clamped_above_max(self):
        # dropout max je 0.4
        result = self.validate({"dropout": 0.9})
        self.assertAlmostEqual(result["dropout"], 0.4)

    def test_range_value_clamped_below_min(self):
        # learning_rate min je 1e-4
        result = self.validate({"learning_rate": 0.00001})
        self.assertAlmostEqual(result["learning_rate"], 1e-4)

    def test_missing_param_skipped(self):
        # Samo hidden_size — ostali se preskačaju bez greške
        result = self.validate({"hidden_size": 128})
        self.assertIn("hidden_size", result)
        self.assertNotIn("num_layers", result)


# ---------------------------------------------------------------------------
# Test: node_decision
# ---------------------------------------------------------------------------

class TestNodeDecision(unittest.TestCase):
    """Provera deterministicke decision logike bez LLM-a."""

    def setUp(self):
        from langraph import node_decision
        self.node = node_decision

    def _run(self, state):
        return self.node(state)

    def test_suggest_only_when_mcc_high(self):
        state = _make_state(mcc=0.92)
        state["regression_detected"] = False
        result = self._run(state)
        self.assertEqual(result["decision"], "SUGGEST_ONLY")

    def test_scan_first_when_mcc_mid(self):
        state = _make_state(mcc=0.80)
        state["regression_detected"] = False
        result = self._run(state)
        self.assertEqual(result["decision"], "SCAN_FIRST")

    def test_retrain_when_mcc_low(self):
        state = _make_state(mcc=0.70)
        result = self._run(state)
        self.assertEqual(result["decision"], "RETRAIN")

    def test_scan_first_when_regression_even_if_mcc_ok(self):
        # MCC je 0.87 (iznad SCAN praga) ali ima regresiju
        state = _make_state(mcc=0.87)
        state["regression_detected"] = True
        result = self._run(state)
        self.assertEqual(result["decision"], "SCAN_FIRST")


# ---------------------------------------------------------------------------
# Test: node_human_confirm
# ---------------------------------------------------------------------------

class TestNodeHumanConfirm(unittest.TestCase):
    """Provera da human_confirm ispravno čita korisnički unos."""

    def _run(self, user_input: str) -> dict:
        from langraph import node_human_confirm
        state = _make_state()
        with patch("builtins.input", return_value=user_input):
            return asyncio.run(node_human_confirm(state))

    def test_yes_confirms(self):
        self.assertTrue(self._run("y")["human_confirmed"])

    def test_yes_long_confirms(self):
        self.assertTrue(self._run("yes")["human_confirmed"])

    def test_no_rejects(self):
        self.assertFalse(self._run("n")["human_confirmed"])

    def test_empty_rejects(self):
        # Podrazumevano je N — prazan Enter ne pokreće retrain
        self.assertFalse(self._run("")["human_confirmed"])

    def test_random_string_rejects(self):
        self.assertFalse(self._run("maybe")["human_confirmed"])


# ---------------------------------------------------------------------------
# Test: _route_after_confirm
# ---------------------------------------------------------------------------

class TestRouteAfterConfirm(unittest.TestCase):
    """Provera routing logike posle human_confirm."""

    def setUp(self):
        from langraph import _route_after_confirm
        self.route = _route_after_confirm

    def test_confirmed_routes_to_retrain(self):
        state = _make_state(confirmed=True)
        self.assertEqual(self.route(state), "retrain")

    def test_not_confirmed_routes_to_skip(self):
        state = _make_state(confirmed=False)
        self.assertEqual(self.route(state), "skip")


# ---------------------------------------------------------------------------
# Test: _patch_hyperparam_file
# ---------------------------------------------------------------------------

class TestPatchHyperparamFile(unittest.TestCase):
    """
    Provera da _patch_hyperparam_file ispravno menja vrednosti u fajlu
    i pravi backup.
    """

    # Minimalni sadržaj koji imitira strukturu hyperparam.py
    _TEMPLATE = '''\
HYPERPARAMETER_SPACE = {
    "hidden_size":   [64, 128, 256],
    "num_layers":    [1, 2],
    "dropout":       (0.1, 0.4),
    "learning_rate": (1e-4, 1e-2),
    "seq_len":       [20, 30, 50],
}
'''

    def setUp(self):
        from langraph import _patch_hyperparam_file
        self.patch_fn = _patch_hyperparam_file
        # Privremeni direktorijum za svaki test
        self.tmp = tempfile.mkdtemp()
        self.hp_path = Path(self.tmp) / "hyperparam.py"
        self.hp_path.write_text(self._TEMPLATE, encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_backup_created(self):
        self.patch_fn(self.hp_path, {"hidden_size": 256})
        backup = self.hp_path.with_suffix(".py.bak")
        self.assertTrue(backup.exists())

    def test_choice_param_patched(self):
        self.patch_fn(self.hp_path, {"hidden_size": 64})
        content = self.hp_path.read_text()
        self.assertIn('"hidden_size":   [64]', content)

    def test_range_param_patched(self):
        self.patch_fn(self.hp_path, {"dropout": 0.25})
        content = self.hp_path.read_text()
        self.assertIn('"dropout":       (0.25, 0.25)', content)

    def test_unrelated_lines_unchanged(self):
        self.patch_fn(self.hp_path, {"hidden_size": 128})
        content = self.hp_path.read_text()
        # Ostale linije ne smeju biti izmenjene
        self.assertIn('"num_layers":    [1, 2]', content)
        self.assertIn('"seq_len":       [20, 30, 50]', content)


# ---------------------------------------------------------------------------
# Test: node_trigger_retrain (mock subprocess)
# ---------------------------------------------------------------------------

class TestNodeTriggerRetrain(unittest.TestCase):
    """
    Provera da node_trigger_retrain ispravno konstruiše komandu i
    pokreće subprocess. torch_nn.py i hyperparam.py se mocku-ju.
    """

    def _run(self, mock_proc_pid=12345, torch_found=True):
        from langraph import node_trigger_retrain

        state = _make_state(confirmed=True)

        # Mock za pronalazak fajlova u sistemu
        fake_torch  = Path("torch_nn.py")
        fake_hp     = Path("hyperparam.py")

        def fake_find(filename, root):
            if filename == "torch_nn.py":
                return fake_torch if torch_found else None
            if filename == "hyperparam.py":
                return fake_hp
            return None

        mock_proc      = MagicMock()
        mock_proc.pid  = mock_proc_pid

        with patch("langraph._find_file_recursive", side_effect=fake_find), \
             patch("langraph._patch_hyperparam_file"),                       \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            result = asyncio.run(node_trigger_retrain(state))

        return result, mock_popen

    def test_retrain_triggered_true_on_success(self):
        result, _ = self._run()
        self.assertTrue(result["retrain_triggered"])

    def test_command_contains_torch_nn(self):
        result, _ = self._run()
        self.assertIn("torch_nn.py", result["retrain_command"])

    def test_popen_called_once(self):
        _, mock_popen = self._run()
        mock_popen.assert_called_once()

    def test_retrain_false_when_torch_not_found(self):
        result, mock_popen = self._run(torch_found=False)
        self.assertFalse(result["retrain_triggered"])
        mock_popen.assert_not_called()


# ---------------------------------------------------------------------------
# Pokretanje
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  LangGraph Retrain — Unit Tests (Korak 4)")
    print("=" * 65)
    unittest.main(verbosity=2)