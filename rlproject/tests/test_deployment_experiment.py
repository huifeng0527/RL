import importlib.util
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_SRC = REPO_ROOT / "rlproject" / "src"
for path in (REPO_ROOT, DEPLOYMENT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from deployment_metrics import compute_fixed_horizon_tiz
import record_deployment_chase as rollout
import run_deployment_chase_batch as batch


class VirtualHandExecutionTests(unittest.TestCase):
    def test_zero_delay_returns_current_intent(self):
        intent = np.array([0.2, -0.1], dtype=np.float32)
        delayed = rollout.apply_virtual_hand_delay(intent, None)
        np.testing.assert_allclose(delayed, intent)

    def test_three_frame_delay_is_exact(self):
        delay_buffer = rollout.make_virtual_hand_delay_buffer(3)
        commands = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([2.0, 0.0], dtype=np.float32),
            np.array([3.0, 0.0], dtype=np.float32),
            np.array([4.0, 0.0], dtype=np.float32),
        ]
        outputs = [
            rollout.apply_virtual_hand_delay(command, delay_buffer)
            for command in commands
        ]
        np.testing.assert_allclose(outputs[0], [0.0, 0.0])
        np.testing.assert_allclose(outputs[1], [0.0, 0.0])
        np.testing.assert_allclose(outputs[2], [0.0, 0.0])
        np.testing.assert_allclose(outputs[3], commands[0])

    def test_alpha_weights_current_delayed_intent(self):
        executed, diagnostics = rollout.apply_virtual_hand_execution(
            hand_action=np.array([1.0, 0.0], dtype=np.float32),
            stride_hand=1.0,
            last_hand_actual_move=np.array([0.2, 0.0], dtype=np.float32),
            smoothing_alpha=0.5,
            delay_buffer=None,
        )
        np.testing.assert_allclose(diagnostics["smoothed_move"], [0.6, 0.0])
        np.testing.assert_allclose(executed, [0.6, 0.0])

    def test_acceleration_limit_remains_active(self):
        last_move = np.array([-1.0, 0.0], dtype=np.float32)
        executed, diagnostics = rollout.apply_virtual_hand_execution(
            hand_action=np.array([1.0, 1.0], dtype=np.float32),
            stride_hand=0.1,
            last_hand_actual_move=last_move,
            smoothing_alpha=1.0,
            delay_buffer=None,
        )
        self.assertTrue(diagnostics["accel_clipped"])
        self.assertAlmostEqual(
            float(np.linalg.norm(executed - last_move)),
            0.15,
            places=6,
        )


class FixedHorizonTizTests(unittest.TestCase):
    def test_timeout_extends_last_state_to_horizon(self):
        metrics = compute_fixed_horizon_tiz(
            [
                {"t_task_s": 0.0, "in_zpd": True},
                {"t_task_s": 30.0, "in_zpd": True},
                {"t_task_s": 59.95, "in_zpd": True},
            ],
            duration_target_s=60.0,
            done_reason="timeout",
        )
        self.assertAlmostEqual(metrics["zpd_time_s"], 60.0)
        self.assertAlmostEqual(metrics["tiz_fixed_horizon_fraction"], 1.0)

    def test_early_catch_gets_no_unused_horizon_credit(self):
        metrics = compute_fixed_horizon_tiz(
            [
                {"t_task_s": 0.0, "in_zpd": True},
                {"t_task_s": 30.0, "in_zpd": False},
            ],
            duration_target_s=60.0,
            done_reason="caught",
        )
        self.assertAlmostEqual(metrics["zpd_time_s"], 30.0)
        self.assertAlmostEqual(metrics["tiz_fixed_horizon_fraction"], 0.5)

    def test_empty_rows_return_zero(self):
        metrics = compute_fixed_horizon_tiz([], 60.0, "caught")
        self.assertEqual(metrics["zpd_time_s"], 0.0)
        self.assertEqual(metrics["tiz_fixed_horizon_fraction"], 0.0)


class RolloutLoggerTests(unittest.TestCase):
    def test_summary_contains_fixed_horizon_tiz_and_new_dr_fields(self):
        logger_path = (
            DEPLOYMENT_SRC / "callbacks" / "deployment_rollout_logger.py"
        )
        spec = importlib.util.spec_from_file_location(
            "deployment_rollout_logger_test",
            logger_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as out_root:
            logger = module.DeploymentRolloutLogger(
                out_root=out_root,
                metadata={"duration_target_s": 60.0},
                zpd_low_cm=3.5,
                zpd_high_cm=5.5,
            )
            logger.record_step({
                "step": 0,
                "t_task_s": 0.0,
                "in_zpd": True,
                "virtual_hand_smoothing_alpha": 0.7,
                "virtual_hand_delay_frames": 2,
            })
            logger.record_step({
                "step": 1,
                "t_task_s": 30.0,
                "in_zpd": False,
                "virtual_hand_smoothing_alpha": 0.7,
                "virtual_hand_delay_frames": 2,
            })
            summary = logger.close(done_reason="caught")
            self.assertAlmostEqual(summary["zpd_time_s"], 30.0)
            self.assertAlmostEqual(summary["tiz_fixed_horizon_fraction"], 0.5)
            header = (logger.rollout_dir / "timeseries.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertIn("virtual_hand_smoothing_alpha", header)
            self.assertIn("virtual_hand_delay_frames", header)
            self.assertIn("virtual_hand_delayed_dx_cm", header)
            self.assertIn("virtual_hand_smoothed_dx_cm", header)


class LogFormattingTests(unittest.TestCase):
    def test_rollout_configuration_names_controller_and_strides(self):
        args = SimpleNamespace(
            controller="cv_mpc",
            stride=0.35,
            max_step=0.60,
            hand_source="virtual",
            hand_alpha=0.7,
            hand_delay_frames=2,
            control_hz=20.0,
            duration=60.0,
            catch_distance=1.5,
            seed=7,
        )
        text = rollout.format_rollout_configuration(
            args,
            virtual_hand_stride=0.45,
        )
        self.assertIn("Controller   : CV-MPC", text)
        self.assertIn("Robot stride: 0.350 cm/action", text)
        self.assertIn("Hand stride : 0.450 cm/action", text)
        self.assertIn("alpha=0.700", text)
        self.assertIn("delay=2 frame(s)", text)


class BatchGenerationTests(unittest.TestCase):
    @staticmethod
    def make_args():
        return SimpleNamespace(
            subject_prefix="virtual_hand_delay_alpha",
            trials=40,
            batch_seed=20260820,
            seconds=60.0,
            stride=0.35,
            max_step=0.60,
            catch_distance=1.5,
            inter_run_delay=3.0,
            save_video=False,
            no_display=True,
        )

    def test_forty_matched_configurations_generate_eighty_runs(self):
        manifest = batch.build_manifest(
            self.make_args(),
            Path("C:/temporary/deployment_batch_test"),
        )
        self.assertEqual(len(manifest["configurations"]), 40)
        self.assertEqual(len(manifest["runs"]), 80)
        self.assertEqual(
            Counter(
                config["virtual_hand_delay_frames"]
                for config in manifest["configurations"]
            ),
            Counter({0: 10, 1: 10, 2: 10, 3: 10}),
        )
        self.assertTrue(all(
            0.5 <= config["virtual_hand_smoothing_alpha"] < 0.9
            for config in manifest["configurations"]
        ))

        by_pair = defaultdict(list)
        first_controller_counts = Counter()
        for run in manifest["runs"]:
            by_pair[run["pair_id"]].append(run)
            if run["order_in_pair"] == 1:
                first_controller_counts[run["controller"]] += 1
            self.assertIn("--hand-alpha", run["command"])
            self.assertIn("--hand-delay-frames", run["command"])

        self.assertEqual(first_controller_counts, Counter({"league": 20, "cv_mpc": 20}))
        for pair_runs in by_pair.values():
            self.assertEqual({run["controller"] for run in pair_runs}, {"league", "cv_mpc"})
            self.assertEqual(len(pair_runs), 2)
            for field in (
                "internal_seed",
                "hand_stride_cm",
                "virtual_hand_smoothing_alpha",
                "virtual_hand_delay_frames",
            ):
                self.assertEqual(pair_runs[0][field], pair_runs[1][field])

    def test_batch_banner_shows_algorithm_and_both_strides(self):
        manifest = batch.build_manifest(
            self.make_args(),
            Path("C:/temporary/deployment_batch_test"),
        )
        text = batch.format_run_banner(
            manifest["runs"][0],
            run_number=1,
            total_runs=80,
            manifest=manifest,
        )
        self.assertIn("PHYSICAL ROLLOUT 01/80", text)
        self.assertIn("Controller   : League RL", text)
        self.assertIn("Robot stride: 0.350 cm/action", text)
        self.assertIn("Hand stride :", text)
        self.assertIn("Hand DR     : alpha=", text)

    def test_result_matching_rejects_alpha_and_delay_mismatch(self):
        manifest = batch.build_manifest(
            self.make_args(),
            Path("C:/temporary/deployment_batch_test"),
        )
        run = manifest["runs"][0]
        row = {
            "run_id": run["run_id"],
            "pair_id": run["pair_id"],
            "trial_index": run["trial_index"],
            "internal_seed": run["internal_seed"],
            "hand_stride_cm": run["hand_stride_cm"],
            "virtual_hand_smoothing_alpha": run["virtual_hand_smoothing_alpha"],
            "virtual_hand_delay_frames": run["virtual_hand_delay_frames"],
            "order_in_pair": run["order_in_pair"],
            "controller": run["controller"],
        }
        self.assertTrue(batch.result_matches_run(row, run))

        wrong_alpha = dict(row)
        wrong_alpha["virtual_hand_smoothing_alpha"] += 0.01
        self.assertFalse(batch.result_matches_run(wrong_alpha, run))

        wrong_delay = dict(row)
        wrong_delay["virtual_hand_delay_frames"] = (
            int(row["virtual_hand_delay_frames"]) + 1
        ) % 4
        self.assertFalse(batch.result_matches_run(wrong_delay, run))


if __name__ == "__main__":
    unittest.main()
