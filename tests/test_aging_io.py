"""Analysis I/O: campaign-aware spec building (analysis/aging_io.py).

Covers the F6 fix: specs_from_campaign must accept the extension campaign's
OBJECT cell entries ({id, yaml, ...}), not just legacy path strings, and must
derive the proc-series prefix authoritatively from each run's manifest
(agg_<engine_group> for multi-process cells), falling back to the cell yaml.

Run: python3 -m unittest tests.test_aging_io
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis"))

import aging_io  # noqa: E402


def _write_extension_campaign(tmp: Path) -> Path:
    (tmp / "cells").mkdir(parents=True, exist_ok=True)
    # Single-process cell (proc.label) and a multi-process cell (components).
    (tmp / "cells" / "val_vllm.yaml").write_text(
        "cell_id: val_vllm\n"
        "monitors:\n"
        "  proc:\n"
        "    label: vllm_standalone\n"
    )
    (tmp / "cells" / "val_dynamo.yaml").write_text(
        "cell_id: val_dynamo\n"
        "monitors:\n"
        "  components:\n"
        "    engine_group: dynamo\n"
    )
    y = tmp / "campaign.yaml"
    y.write_text(
        "campaign_id: extension_test\n"
        "mode: serial\n"
        f"runs_root: {tmp / 'runs'}\n"
        "replicas_per_cell: 1\n"
        "cells:\n"
        "  - id: val_vllm\n"
        "    yaml: cells/val_vllm.yaml\n"
        "    calibration_required: false\n"
        "  - id: val_dynamo\n"
        "    yaml: cells/val_dynamo.yaml\n"
        "    calibration_required: false\n"
    )
    return y


class NormalizeCells(unittest.TestCase):
    def test_string_and_object_entries(self):
        norm = aging_io.normalize_campaign_cells(
            {"cells": ["cells/a.yaml", {"id": "b", "yaml": "cells/b.yaml"}]}
        )
        self.assertEqual(norm[0], {"id": None, "yaml": "cells/a.yaml"})
        self.assertEqual(norm[1], {"id": "b", "yaml": "cells/b.yaml"})

    def test_object_entry_without_yaml_raises(self):
        with self.assertRaises(ValueError):
            aging_io.normalize_campaign_cells({"cells": [{"id": "b"}]})


class ProcPrefixFromCellDoc(unittest.TestCase):
    def test_multiprocess_components_aggregate(self):
        doc = {"monitors": {"components": {"engine_group": "dynamo"}}}
        self.assertEqual(aging_io.proc_prefix_from_cell_doc(doc), "agg_dynamo")

    def test_components_without_group_defaults_engine(self):
        doc = {"monitors": {"components": {"n_prefill": 1}}}
        self.assertEqual(aging_io.proc_prefix_from_cell_doc(doc), "agg_engine")

    def test_single_process_uses_proc_label(self):
        doc = {"monitors": {"proc": {"label": "vllm_standalone"}}}
        self.assertEqual(aging_io.proc_prefix_from_cell_doc(doc), "vllm_standalone")


class SpecsFromExtensionCampaign(unittest.TestCase):
    def test_object_cells_build_specs_without_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            y = _write_extension_campaign(Path(tmp))
            specs = aging_io.specs_from_campaign(y)
            run_ids = [s.run_dir.name for s in specs]
            self.assertEqual(
                run_ids,
                ["extension_test_val_vllm_r01", "extension_test_val_dynamo_r01"],
            )
            by_cell = {s.cell_id: s for s in specs}
            # No manifest on disk yet -> fall back to the cell-yaml derivation.
            self.assertEqual(by_cell["val_vllm"].proc_prefix, "vllm_standalone")
            self.assertEqual(by_cell["val_dynamo"].proc_prefix, "agg_dynamo")

    def test_manifest_proc_prefix_is_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            y = _write_extension_campaign(tmp)
            # launch_cell recorded a specific engine-aggregate prefix for the run.
            run_dir = tmp / "runs" / "extension_test_val_dynamo_r01"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                json.dumps({"cell_id": "val_dynamo", "proc_prefix": "agg_dynamo_leader"})
            )
            specs = aging_io.specs_from_campaign(y)
            by_cell = {s.cell_id: s for s in specs}
            self.assertEqual(by_cell["val_dynamo"].proc_prefix, "agg_dynamo_leader")

    def test_cell_filter_applies(self):
        with tempfile.TemporaryDirectory() as tmp:
            y = _write_extension_campaign(Path(tmp))
            specs = aging_io.specs_from_campaign(y, cells={"val_vllm"})
            self.assertEqual([s.cell_id for s in specs], ["val_vllm"])


class MinimalCampaignParser(unittest.TestCase):
    """The PyYAML-free fallback parser must also understand object cell entries."""

    def test_object_entries_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            y = Path(tmp) / "campaign.yaml"
            y.write_text(
                "campaign_id: extc\n"
                "runs_root: /tmp/runs\n"
                "replicas_per_cell: 1\n"
                "cells:\n"
                "  - id: val_vllm\n"
                "    yaml: cells/val_vllm.yaml\n"
                "    calibration_required: false\n"
                "  - id: val_dynamo\n"
                "    yaml: cells/val_dynamo.yaml\n"
            )
            data = aging_io.parse_campaign_yaml_minimal(y)
            self.assertEqual(data["campaign_id"], "extc")
            norm = aging_io.normalize_campaign_cells(data)
            self.assertEqual(
                [(e["id"], e["yaml"]) for e in norm],
                [("val_vllm", "cells/val_vllm.yaml"),
                 ("val_dynamo", "cells/val_dynamo.yaml")],
            )

    def test_legacy_string_entries_still_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            y = Path(tmp) / "campaign.yaml"
            y.write_text(
                "campaign_id: legacy\n"
                "runs_root: /tmp/runs\n"
                "cells:\n"
                "  - cells/e1.yaml\n"
                "  - cells/a1.yaml\n"
            )
            data = aging_io.parse_campaign_yaml_minimal(y)
            self.assertEqual(data["cells"], ["cells/e1.yaml", "cells/a1.yaml"])


if __name__ == "__main__":
    unittest.main()
