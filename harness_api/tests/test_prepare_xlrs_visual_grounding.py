import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prepare_xlrs_visual_grounding_test",
    PROJECT / "scripts" / "prepare_xlrs_visual_grounding.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(
    importlib.util.find_spec("pyarrow"), "requires isolated packed-requirements"
)
class PrepareXLRSVisualGroundingTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa

        self.pa = pa
        self.png = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d4944415478da63fcffff3f030005fe02fea72d4e4b0000000049454e44ae426082"
        )
        self.source_row = {
            "image_width": 1.0,
            "image_height": 1.0,
            "question_id": "Visual grounding/test/00001",
            "image": {"bytes": self.png, "path": "source-image.png"},
            "question": (
                "Given an 1 x 1 image, identify the bounding box. Description: "
                "Locate the bright target."
            ),
            "answer": "Locate the bright target.",
            "bbox": [0.25, 0.25, 0.75, 0.75],
            "category": "Visual grounding/test",
        }
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "runtime" / "grounding"
        schema = self.pa.schema(
            [
                self.pa.field("image_width", self.pa.float64()),
                self.pa.field("image_height", self.pa.float64()),
                self.pa.field("question_id", self.pa.string()),
                self.pa.field(
                    "image",
                    self.pa.struct(
                        [("bytes", self.pa.binary()), ("path", self.pa.string())]
                    ),
                ),
                self.pa.field("question", self.pa.string()),
                self.pa.field("answer", self.pa.string()),
                self.pa.field("bbox", self.pa.list_(self.pa.float64())),
                self.pa.field("category", self.pa.string()),
            ]
        )
        row = self.source_row
        path = self.source / "data-00000-of-00001.arrow"
        with self.pa.OSFile(str(path), "wb") as sink, self.pa.ipc.new_stream(
            sink, schema
        ) as writer:
            writer.write_table(self.pa.Table.from_pylist([row], schema=schema))

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_separates_public_prompt_from_hidden_truth(self):
        receipt = MODULE.prepare(self.source, self.output, 1)
        self.assertEqual(receipt["sample_count"], 1)
        sample = receipt["samples"][0]
        task_dir = self.output / "tasks" / sample["task_id"]
        task = json.loads((task_dir / "task.json").read_text())
        evaluator = json.loads((task_dir / "evaluator.json").read_text())
        assets = json.loads((task_dir / "assets.json").read_text())
        inputs = json.loads((self.output / "inputs.json").read_text())
        digest = hashlib.sha256(self.png).hexdigest()
        self.assertEqual(task["prompt"], (
            "Locate the target described in the public question. First crop an AOI "
            "that contains the target, save that frozen crop as evidence, then submit "
            "the target bbox in normalized image coordinates [xmin,ymin,xmax,ymax]. "
            "Locate the bright target."
        ))
        self.assertNotIn("Locate the bright target", json.dumps(evaluator))
        self.assertEqual(evaluator["config"]["expected_bbox"], [0.25, 0.25, 0.75, 0.75])
        self.assertEqual(assets[0]["media_type"], "image/png")
        self.assertEqual(inputs[assets[0]["asset_id"]]["sha256"], digest)
        self.assertEqual(
            hashlib.sha256(
                (self.output / "inputs" / (digest + ".png")).read_bytes()
            ).hexdigest(),
            digest,
        )

    def test_rejects_declared_dimension_mismatch(self):
        invalid = self.root / "invalid-source"
        invalid.mkdir()
        self._write_shard(invalid, {**self.source_row, "image_width": 9.0})
        with self.assertRaisesRegex(ValueError, "insufficient valid"):
            MODULE.prepare(invalid, self.output / "invalid", 1)

    def _write_shard(self, directory: Path, row: dict) -> None:
        schema = self._schema()
        path = directory / "data-00000-of-00001.arrow"
        table = self.pa.Table.from_pylist([row], schema=schema)
        with self.pa.OSFile(str(path), "wb") as sink, self.pa.ipc.new_stream(
            sink, schema
        ) as writer:
            writer.write_table(table)

    def _schema(self):
        return self.pa.schema(
            [
                self.pa.field("image_width", self.pa.float64()),
                self.pa.field("image_height", self.pa.float64()),
                self.pa.field("question_id", self.pa.string()),
                self.pa.field(
                    "image",
                    self.pa.struct(
                        [("bytes", self.pa.binary()), ("path", self.pa.string())]
                    ),
                ),
                self.pa.field("question", self.pa.string()),
                self.pa.field("answer", self.pa.string()),
                self.pa.field("bbox", self.pa.list_(self.pa.float64())),
                self.pa.field("category", self.pa.string()),
            ]
        )


if __name__ == "__main__":
    unittest.main()
