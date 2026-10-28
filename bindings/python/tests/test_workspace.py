"""Unit tests for Argon Engine Google Workspace & GSuite integration."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from argon_engine import (
    ArgonEngine,
    GoogleDocsReportGenerator,
    GoogleDriveModelArchive,
    GoogleSheetsTelemetrySync,
)


def test_google_sheets_telemetry_sync_rows_and_formulas():
    sync = GoogleSheetsTelemetrySync(title="Test Flight Telemetry")
    assert sync.row_count == 0

    sync.record_tick(sequence_id=1, mean_goodness=0.85, output_l2_norm=1.2, queue_depth=0, phase_polarity=1.0)
    sync.record_tick(sequence_id=2, mean_goodness=0.92, output_l2_norm=1.1, queue_depth=1, phase_polarity=1.0)
    sync.record_tick(sequence_id=3, mean_goodness=0.45, output_l2_norm=0.9, queue_depth=0, phase_polarity=-1.0)

    assert sync.row_count == 3

    csv_text = sync.to_csv_string(include_summary_formulas=True)
    lines = [line.strip() for line in csv_text.strip().split("\n") if line.strip()]

    # Header check
    assert lines[0] == "Timestamp_ISO,Sequence_ID,Mean_Goodness,Output_L2_Norm,Queue_Depth,Phase_Polarity,Status"
    # Row checks
    assert "1,0.85,1.2,0,1.0,OK" in lines[1]
    assert "2,0.92,1.1,1,1.0,OK" in lines[2]
    assert "3,0.45,0.9,0,-1.0,OK" in lines[3]

    # Google Sheets formula checks
    assert "SUMMARY_AVERAGE,=COUNT(B2:B4),=AVERAGE(C2:C4),=AVERAGE(D2:D4),=AVERAGE(E2:E4),=AVERAGE(F2:F4),COMPUTED" in csv_text
    assert "SUMMARY_MAX,=MAX(B2:B4),=MAX(C2:C4),=MAX(D2:D4),=MAX(E2:E4),-,COMPUTED" in csv_text

    # Google Sheets API payload structure check
    payload = sync.to_sheets_payload()
    assert payload["range"] == "Sheet1!A1:G4"
    assert payload["majorDimension"] == "ROWS"
    assert len(payload["values"]) == 4  # Header + 3 rows


def test_google_docs_report_generation():
    report = GoogleDocsReportGenerator.generate_markdown_report(
        engine_topology={"num_layers": 2, "tile_dim": 8, "learning_rate": 0.02},
        initial_checksum=0x123456789ABCDEF0,
        final_checksum=0xFEDCBA9876543210,
        telemetry_summary={
            "total_ticks": 1000,
            "mean_goodness": 1.452,
            "output_l2_norm": 1.05,
            "p99_latency_ns": 480.0,
        },
        hardware_info={"platform": "NVIDIA SM80+ Persistent SRAM", "parity_verified": True},
    )

    assert "# Argon Engine Execution & Experiment Report" in report
    assert "0x123456789abcdef0" in report
    assert "0xfedcba9876543210" in report
    assert "p99 Tick Latency" in report
    assert "480.0 ns" in report
    assert "✅ PASS" in report


def test_google_drive_model_archive_packaging():
    with tempfile.TemporaryDirectory() as tmpdir:
        weights = np.ones((128,), dtype=np.float32) * 0.25
        sync = GoogleSheetsTelemetrySync()
        sync.record_tick(sequence_id=0, mean_goodness=1.0, output_l2_norm=1.0)

        bundle = GoogleDriveModelArchive.package_bundle(
            output_directory=tmpdir,
            weights=weights,
            topology={"num_layers": 2, "tile_dim": 8},
            telemetry_sync=sync,
        )

        assert os.path.exists(bundle["weights"])
        assert os.path.exists(bundle["metadata"])
        assert os.path.exists(bundle["telemetry_csv"])

        # Recover weights from archive
        loaded_weights = np.load(bundle["weights"])
        np.testing.assert_array_equal(loaded_weights, weights)

        # Inspect metadata JSON
        meta = json.loads(Path(bundle["metadata"]).read_text(encoding="utf-8"))
        assert meta["argon_version"] == "0.1.0"
        assert meta["weight_elements"] == 128


def test_argon_engine_end_to_end_workspace_export():
    with ArgonEngine(num_layers=2, tile_dim=8) as engine:
        sync = GoogleSheetsTelemetrySync(title="Live Hardware Test")
        initial_checksum = engine.weight_checksum()

        inputs = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        out_buf = np.zeros(8, dtype=np.float32)

        for step in range(5):
            _, telem = engine.step(inputs, polarity=1.0, output=out_buf)
            sync.record_engine_telemetry(telem, phase_polarity=1.0)

        assert sync.row_count == 5
        final_checksum = engine.weight_checksum()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "live_telemetry.csv"
            sync.export_csv(csv_path)
            assert csv_path.exists()
            assert "SUMMARY_AVERAGE" in csv_path.read_text(encoding="utf-8")

            report_path = Path(tmpdir) / "session_report.md"
            report_text = GoogleDocsReportGenerator.generate_markdown_report(
                engine_topology={"num_layers": 2, "tile_dim": 8},
                initial_checksum=initial_checksum,
                final_checksum=final_checksum,
                telemetry_summary={
                    "total_ticks": 5,
                    "mean_goodness": telem.mean_goodness,
                    "output_l2_norm": telem.output_l2_norm,
                    "p99_latency_ns": 520.0,
                },
            )
            GoogleDocsReportGenerator.export_report(report_path, report_text)
            assert report_path.exists()
            assert "Argon Engine Execution & Experiment Report" in report_path.read_text(encoding="utf-8")

