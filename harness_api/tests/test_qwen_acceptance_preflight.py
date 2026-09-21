import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("qwen_preflight", PROJECT / "scripts" / "preflight_qwen_acceptance.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QwenPreflightTests(unittest.TestCase):
    def test_matrix_and_runtime_files_are_ready(self):
        matrix = MODULE.matrix_status()
        self.assertEqual(matrix["ready_datasets"], 11)
        self.assertEqual(matrix["total_samples"], 22)
        self.assertEqual(matrix["pending_datasets"], [])
        runtime = MODULE.runtime_file_status()
        self.assertTrue(all(value["exists"] for value in runtime.values()))

    def test_gpu_and_slurm_fail_closed(self):
        with patch.object(MODULE, "command", side_effect=[
            (0, "0, 0 MiB, 81920 MiB, 0 %\n1, 0 MiB, 81920 MiB, 0 %"),
            (0, "NODE STATE REASON\nlamda12 drained* reason"),
            (0, ""),
        ]):
            gpu = MODULE.gpu_status()
            slurm = MODULE.slurm_status()
        self.assertEqual(len(gpu["gpus"]), 2)
        self.assertFalse(slurm["allocation_ready"])

    def test_model_completeness_and_environment_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in MODULE.MODEL_FILES:
                (root / name).write_bytes(name.encode())
            (root / "model.safetensors-00001-of-00001.safetensors").write_bytes(b"weights")
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model.safetensors-00001-of-00001.safetensors"}}))
            model = MODULE.model_status(root)
            self.assertTrue(model["weight_files_complete"])
        self.assertFalse(MODULE.environment_status(Path("/missing/python"))["available"])


if __name__ == "__main__":
    unittest.main()
