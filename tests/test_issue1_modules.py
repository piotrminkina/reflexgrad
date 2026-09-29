"""Regression tests for issue #1: `enhanced_logging` and `universal_memory_system`.

Run from the repository root:  python -m unittest discover -s tests -v

Requires only numpy. No model/API calls, no GPU, no ALFWorld data.

Importing `universal_memory_system` instantiates a module-level singleton that
creates and reads ./universal_memory/ in the current working directory, so every
test here imports the modules from inside a throwaway working directory.
"""
import ast
import importlib
import inspect
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CALLERS = ["alfworld_trial.py", "reflexgrad_trial.py", "generate_reflections.py"]


def _fresh_import(name, workdir):
    """Import `name` from the repo while cwd is `workdir`, bypassing any cache."""
    sys.modules.pop(name, None)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    old = os.getcwd()
    os.chdir(workdir)
    try:
        return importlib.import_module(name)
    finally:
        os.chdir(old)


def _call_kwargs(receiver):
    """Every `receiver.method(kw=..., ...)` call in the public callers, via AST."""
    found = []
    for fname in CALLERS:
        tree = ast.parse((REPO / fname).read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == receiver):
                found.append((fname, node.lineno, node.func.attr,
                              [k.arg for k in node.keywords], len(node.args)))
    return found


class ImportResolution(unittest.TestCase):
    def test_both_modules_resolve_inside_this_checkout(self):
        with tempfile.TemporaryDirectory() as wd:
            for name in ("enhanced_logging", "universal_memory_system"):
                mod = _fresh_import(name, wd)
                self.assertEqual(Path(mod.__file__).resolve().parent, REPO, name)
            from enhanced_logging import ComprehensiveLogger  # noqa: F401
            from universal_memory_system import universal_memory  # noqa: F401


class CallSiteInterfaces(unittest.TestCase):
    """Every call the released code makes must bind to a real method signature."""

    def _check(self, receiver, cls):
        calls = _call_kwargs(receiver)
        self.assertTrue(calls, f"no call sites found for {receiver}")
        for fname, line, meth, kws, npos in calls:
            self.assertTrue(hasattr(cls, meth), f"{fname}:{line} calls missing {receiver}.{meth}")
            sig = inspect.signature(getattr(cls, meth))
            try:
                sig.bind(object(), *([None] * npos), **{k: None for k in kws})
            except TypeError as e:
                self.fail(f"{fname}:{line} {receiver}.{meth}: {e}")

    def test_comprehensive_logger_calls_bind(self):
        with tempfile.TemporaryDirectory() as wd:
            self._check("comprehensive_logger", _fresh_import("enhanced_logging", wd).ComprehensiveLogger)

    def test_universal_memory_calls_bind(self):
        with tempfile.TemporaryDirectory() as wd:
            self._check("universal_memory", _fresh_import("universal_memory_system", wd).UniversalMemorySystem)


class ComprehensiveLoggerBehavior(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "trial"
        self.log = _fresh_import("enhanced_logging", self.tmp.name).ComprehensiveLogger(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def _jsonl(self, name):
        return [json.loads(l) for l in (self.dir / name).read_text().splitlines()]

    def test_step_gradient_format_and_truncation(self):
        self.log.log_step_gradient(3, 7, {
            "action": "go to desk 1", "progress_status": "ADVANCING",
            "semantic_state_before": "x" * 500, "semantic_state_after": "y",
            "prerequisites": {"missing": ["lamp"]}, "task_progress": {"remaining": ["look at lamp"]},
        })
        (e,) = self._jsonl("step_gradients.jsonl")
        self.assertEqual((e["env_id"], e["step"], e["action"]), (3, 7, "go to desk 1"))
        self.assertEqual(len(e["semantic_state_before"]), 200)
        self.assertEqual(e["prerequisites_missing"], ["lamp"])
        self.assertEqual(e["task_remaining"], ["look at lamp"])
        self.assertEqual(e["guidance_source"], "unknown")

    def test_empty_gradient_uses_documented_defaults(self):
        self.log.log_step_gradient(0, 0, {})
        (e,) = self._jsonl("step_gradients.jsonl")
        self.assertEqual((e["action"], e["progress_status"]), ("N/A", "EXPLORING"))

    def test_reflexion_memory_and_episodic_outputs(self):
        self.log.log_step_reflexion(1, 2, "take mug", "it worked", True)
        self.log.log_memory_update(1, "global_memory_added", {"k": "v" * 900})
        self.log.log_episodic_reflection(1, "put a mug", "REFLECTION", {"a": "g1", "structured_insights": "hidden"})
        self.assertEqual(self._jsonl("step_reflexions.jsonl")[0]["success"], True)
        self.assertLessEqual(len(self._jsonl("memory_evolution.jsonl")[0]["content"]), 500)
        text = (self.dir / "episodic_reflections.txt").read_text()
        self.assertIn("Task: put a mug", text)
        self.assertIn("a: g1", text)
        self.assertNotIn("hidden", text)  # structured_insights is deliberately excluded

    def test_logs_append_across_instances(self):
        cls = type(self.log)
        cls(self.dir).log_step_reflexion(1, 1, "a", "r", False)
        cls(self.dir).log_step_reflexion(1, 2, "b", "r", False)
        self.assertEqual(len(self._jsonl("step_reflexions.jsonl")), 2)


class UniversalMemoryBehavior(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mod = _fresh_import("universal_memory_system", self.tmp.name)
        self.mem = self.mod.UniversalMemorySystem(memory_dir=str(Path(self.tmp.name) / "m"))

    def tearDown(self):
        self.tmp.cleanup()

    def _interact(self, action, success, before="you are at desk 1", after="you see a lamp",
                  missing=("lamp",), remaining=("lamp",)):
        self.mem.record_semantic_interaction(
            semantic_state_before=before, action=action, action_reasoning="r",
            semantic_state_after=after, prerequisites={"present": [], "missing": list(missing)},
            task_progress={"addressed": [], "remaining": list(remaining)},
            success=success, task="look at lamp", episode_id="ep0")

    def test_singleton_exists_with_expected_state(self):
        self.assertIsInstance(self.mod.universal_memory, self.mod.UniversalMemorySystem)
        self.assertEqual(len(self.mod.universal_memory.state_action_outcomes), 0)

    def test_failed_action_is_recommended_against(self):
        self._interact("use lamp", success=False)
        rec = self.mem.get_semantic_recommendations("s", ["lamp"], ["use lamp", "go to desk 2"])
        self.assertEqual([r["action"] for r in rec["avoid"]], ["use lamp"])
        self.assertEqual(rec["unexplored"], ["go to desk 2"])
        self.assertEqual(self.mem.total_interactions, 1)

    def test_recommendation_shape_when_empty(self):
        rec = self.mem.get_semantic_recommendations("s", [], [])
        self.assertEqual(set(rec), {"strongly_recommended", "previously_succeeded", "avoid", "unexplored"})

    def test_episode_and_sequence_patterns_use_three_tuple_trajectories(self):
        traj = [("go to desk 1", "obs1", 0), ("take mug 1", "obs2", 0), ("use lamp 1", "obs3", 1)]
        self.mem.record_episode(traj, "look at lamp", True)
        self.mem.store_sequence_pattern(traj, True, "look at lamp")
        self.assertEqual((self.mem.successful_episodes, self.mem.failed_episodes), (1, 0))
        self.assertEqual(self.mem.match_sequence_pattern(["go to desk 1", "take mug 1"], "t"), "use lamp 1")
        self.mem.record_episode(traj, "t", False)
        self.assertEqual(self.mem.get_statistics()["total_episodes"], 2)

    def test_store_sequence_pattern_ignores_failures(self):
        self.mem.store_sequence_pattern([("a", "o", 0)], False, "t")
        self.assertEqual(self.mem.sequence_patterns["successful_sequences"], [])

    def test_save_load_roundtrip_persists_documented_fields(self):
        traj = [("a", "o1", 0), ("b", "o2", 0), ("c", "o3", 1)]
        self.mem.record_episode(traj, "t", True)
        self.mem.store_sequence_pattern(traj, True, "t")
        self.mem.save_memory()
        again = self.mod.UniversalMemorySystem(memory_dir=str(self.mem.memory_dir))
        self.assertEqual(again.successful_episodes, 1)
        self.assertEqual(again.match_sequence_pattern(["a", "b"], "t"), "c")

    def test_separate_memory_dirs_are_isolated(self):
        self.mem.record_episode([("a", "o", 0)], "t", True)
        self.mem.save_memory()
        other = self.mod.UniversalMemorySystem(memory_dir=str(Path(self.tmp.name) / "other"))
        self.assertEqual(other.successful_episodes, 0)

    def test_corrupt_pickle_falls_back_to_fresh_memory(self):
        d = Path(self.tmp.name) / "bad"
        d.mkdir()
        (d / "universal_memory.pkl").write_bytes(b"not a pickle")
        mem = self.mod.UniversalMemorySystem(memory_dir=str(d))
        self.assertEqual((mem.total_interactions, mem.successful_episodes), (0, 0))

    def test_determinism_and_exploration_helpers(self):
        self.assertEqual(self.mem.is_action_deterministic("s", "a"), (True, 0.0))
        self.mem.state_action_outcomes["s"]["a"] = [
            {"result_hash": "x", "effectiveness": 0.0}, {"result_hash": "y", "effectiveness": 1.0}]
        det, _ = self.mem.is_action_deterministic("s", "a")
        self.assertFalse(det)
        self.assertTrue(self.mem.should_explore("s", ["a", "b", "c", "d"], set())[0])


if __name__ == "__main__":
    unittest.main()
