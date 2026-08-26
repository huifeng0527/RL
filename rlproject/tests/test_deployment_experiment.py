import importlib.util
import json
import sys
import tempfile
import time
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


class ServoInterpolationTests(unittest.TestCase):
    class FakeRTDEControl:
        pass

    class FakeRobotControl:
        def __init__(self, servo_return=None):
            self.rtde_c = ServoInterpolationTests.FakeRTDEControl()
            self.servo_calls = []
            self.stop_count = 0
            self.servo_return = servo_return

        def servo_robot(self, target_pose, dt, lookahead_time, gain):
            self.servo_calls.append({
                "target_pose": np.asarray(target_pose, dtype=float),
                "dt": float(dt),
                "lookahead_time": float(lookahead_time),
                "gain": int(gain),
            })
            return self.servo_return

        def servo_stop(self):
            self.stop_count += 1

    def test_linear_pose_interpolation_and_endpoint_hold(self):
        start = np.zeros(6, dtype=float)
        target = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            rollout.linear_interpolate_pose(start, target, 0.0),
            start,
        )
        np.testing.assert_allclose(
            rollout.linear_interpolate_pose(start, target, 0.5),
            target * 0.5,
        )
        np.testing.assert_allclose(
            rollout.linear_interpolate_pose(start, target, 2.0),
            target,
        )

    def test_translation_step_limit_preserves_requested_orientation(self):
        previous = np.zeros(6, dtype=float)
        requested = np.array([0.03, 0.04, 0.0, 0.1, 0.2, 0.3])
        command, limited = rollout.limit_pose_translation_step(
            previous,
            requested,
            max_step_m=0.01,
        )
        self.assertTrue(limited)
        self.assertAlmostEqual(np.linalg.norm(command[:3]), 0.01)
        np.testing.assert_allclose(command[3:], requested[3:])

    def test_default_timing_separates_policy_and_servo_rates(self):
        timing = rollout.normalize_servo_timing(
            policy_hz=20.0,
            servo_hz=125.0,
            max_step_cm=1.0,
        )
        self.assertEqual(timing.mode, "interpolated")
        self.assertEqual(timing.trajectory_mode, rollout.SERVO_TRAJECTORY_MODE)
        self.assertEqual(timing.policy_hz, 20.0)
        self.assertEqual(timing.servo_hz, 125.0)
        self.assertAlmostEqual(timing.target_timeout_s, 0.15)
        self.assertGreaterEqual(timing.startup_grace_s, timing.target_timeout_s)
        self.assertLess(timing.lag_warning_cm, timing.lag_hard_limit_cm)
        # One saturated policy step needs max_step_cm per policy period; the
        # per-tick cap carries headroom above that so it never rate-limits the
        # step the interpolation is supposed to land on time.
        saturated_step_per_tick_m = (
            (1.0 / 100.0) * timing.servo_period_s / timing.policy_period_s
        )
        self.assertAlmostEqual(
            timing.max_translation_per_tick_m,
            saturated_step_per_tick_m * timing.speed_headroom,
        )
        self.assertGreater(
            timing.max_translation_per_tick_m,
            saturated_step_per_tick_m,
        )

    def test_legacy_mode_reports_legacy_trajectory(self):
        timing = rollout.normalize_servo_timing(
            policy_hz=20.0,
            servo_mode="legacy",
            servo_hz=125.0,
        )
        self.assertEqual(timing.trajectory_mode, rollout.LEGACY_TRAJECTORY_MODE)
        self.assertEqual(timing.servo_hz, 20.0)

    def test_timing_rejects_inverted_lag_thresholds(self):
        with self.assertRaises(ValueError):
            rollout.normalize_servo_timing(
                policy_hz=20.0,
                servo_hz=125.0,
                lag_warning_cm=3.0,
                lag_hard_limit_cm=1.5,
            )

    def test_follower_lag_allowance_covers_one_period_plus_lookahead(self):
        timing = rollout.normalize_servo_timing(
            policy_hz=20.0,
            servo_hz=125.0,
            max_step_cm=1.0,
            lookahead_time_s=0.1,
        )
        # A saturated step travels max_step_cm per policy period, and the arm
        # only reaches that endpoint one period later plus the lookahead phase
        # lag, so a healthy full-speed step already trails by 3 cm.
        self.assertAlmostEqual(timing.follower_lag_allowance_cm, 3.0)
        tighter = rollout.normalize_servo_timing(
            policy_hz=20.0,
            servo_hz=125.0,
            max_step_cm=1.0,
            lookahead_time_s=0.05,
        )
        self.assertAlmostEqual(tighter.follower_lag_allowance_cm, 2.0)

    def test_lag_guard_ignores_the_structural_follower_lag(self):
        # The defect this guards against: comparing the raw planned-vs-actual
        # gap against the thresholds flags every sustained full-speed chase,
        # because the follower's own phase lag already fills the whole budget.
        timing = rollout.normalize_servo_timing(
            policy_hz=20.0,
            servo_hz=125.0,
            max_step_cm=1.0,
            lookahead_time_s=0.1,
            lag_warning_cm=1.5,
            lag_hard_limit_cm=3.0,
        )
        excess, over_warning, over_hard = rollout.evaluate_lag_guard(
            timing.follower_lag_allowance_cm,
            timing,
        )
        self.assertAlmostEqual(excess, 0.0)
        self.assertFalse(over_warning)
        self.assertFalse(over_hard)

        excess, over_warning, over_hard = rollout.evaluate_lag_guard(
            timing.follower_lag_allowance_cm + 2.0,
            timing,
        )
        self.assertAlmostEqual(excess, 2.0)
        self.assertTrue(over_warning)
        self.assertFalse(over_hard)

        excess, over_warning, over_hard = rollout.evaluate_lag_guard(
            timing.follower_lag_allowance_cm + 4.0,
            timing,
        )
        self.assertTrue(over_warning)
        self.assertTrue(over_hard)

    def test_lag_guard_is_inactive_when_the_anchor_tracks_the_measurement(self):
        timing = rollout.normalize_servo_timing(
            policy_hz=20.0,
            servo_mode="legacy",
            servo_hz=20.0,
            max_step_cm=1.0,
        )
        excess, over_warning, over_hard = rollout.evaluate_lag_guard(
            50.0,
            timing,
            anchored_on_planned_frame=False,
        )
        self.assertGreater(excess, 0.0)
        self.assertFalse(over_warning)
        self.assertFalse(over_hard)

    def test_interpolated_mode_rejects_slower_servo_rate(self):
        with self.assertRaises(ValueError):
            rollout.normalize_servo_timing(
                policy_hz=20.0,
                servo_hz=10.0,
            )

    def test_segment_starts_at_previous_committed_endpoint(self):
        # The defect this guards against: anchoring each segment on the servo
        # loop's instantaneous command pose (which lags the target) makes the
        # commanded frame advance less than one policy step per period, so the
        # arm never reaches the intended speed no matter how fast the servo runs.
        robot = self.FakeRobotControl()
        timing = rollout.normalize_servo_timing(
            policy_hz=20.0,
            servo_hz=200.0,
            target_timeout_s=1.0,
            max_step_cm=1.0,
        )
        thread = rollout.InterpolatedServoThread(
            robot,
            initial_pose=np.zeros(6, dtype=float),
            timing=timing,
        )
        first = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        second = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
        sequence_one, pending_one, reason_one = thread.publish_segment(
            first,
            policy_step=0,
        )
        self.assertEqual(reason_one, "")
        self.assertEqual(sequence_one, 1)
        self.assertEqual(pending_one, 2)
        np.testing.assert_allclose(thread.committed_end_pose(), first)

        # Published before the servo loop ever ran, so no command pose exists yet;
        # the second segment must still start exactly where the first ended.
        sequence_two, _, reason_two = thread.publish_segment(second, policy_step=1)
        self.assertEqual(reason_two, "")
        self.assertEqual(sequence_two, 2)
        np.testing.assert_allclose(thread.committed_end_pose(), second)
        segments = list(thread._segments)
        np.testing.assert_allclose(segments[-1].start_pose, first)
        np.testing.assert_allclose(segments[-2].end_pose, segments[-1].start_pose)
        self.assertGreaterEqual(segments[-1].start_perf, segments[-2].end_perf)
        self.assertAlmostEqual(
            segments[-1].duration_s,
            timing.policy_period_s,
            places=9,
        )

    def test_segment_phase_is_clamped_to_its_own_window(self):
        segment = rollout.ServoSegment(
            start_pose=np.zeros(6, dtype=float),
            end_pose=np.ones(6, dtype=float),
            start_perf=10.0,
            end_perf=10.05,
            policy_step=0,
            sequence=1,
        )
        self.assertEqual(segment.phase_at(9.9), 0.0)
        self.assertAlmostEqual(segment.phase_at(10.025), 0.5)
        self.assertEqual(segment.phase_at(10.2), 1.0)

    def test_publish_rejects_when_segment_queue_saturates(self):
        robot = self.FakeRobotControl()
        timing = rollout.normalize_servo_timing(
            policy_hz=20.0,
            servo_hz=200.0,
            target_timeout_s=1.0,
            max_step_cm=1.0,
        )
        thread = rollout.InterpolatedServoThread(
            robot,
            initial_pose=np.zeros(6, dtype=float),
            timing=timing,
        )
        reasons = []
        for index in range(rollout.SERVO_MAX_PENDING_SEGMENTS + 2):
            _, _, reason = thread.publish_segment(
                np.array([0.001 * (index + 1), 0.0, 0.0, 0.0, 0.0, 0.0]),
                policy_step=index,
            )
            reasons.append(reason)
        self.assertEqual(reasons[0], "")
        self.assertIn("queue_saturated", reasons)
        self.assertEqual(
            len(thread._segments),
            rollout.SERVO_MAX_PENDING_SEGMENTS,
        )
        self.assertGreater(thread.publish_rejected_count, 0)

    def test_servo_thread_publishes_fixed_dt_and_stops(self):
        robot = self.FakeRobotControl()
        timing = rollout.normalize_servo_timing(
            policy_hz=20.0,
            servo_hz=200.0,
            target_timeout_s=1.0,
            max_step_cm=1.0,
        )
        thread = rollout.InterpolatedServoThread(
            robot,
            initial_pose=np.zeros(6, dtype=float),
            timing=timing,
        )
        thread.start()
        thread.publish_segment(
            np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
            policy_step=0,
        )
        time.sleep(0.03)
        thread.stop_and_join("test_complete")
        self.assertGreaterEqual(len(robot.servo_calls), 2)
        self.assertTrue(all(
            abs(call["dt"] - 1.0 / 200.0) < 1e-9
            for call in robot.servo_calls
        ))
        self.assertEqual(robot.stop_count, 1)
        self.assertEqual(thread.snapshot().stop_reason, "test_complete")

    def test_commanded_frame_covers_a_full_step_per_policy_period(self):
        robot = self.FakeRobotControl()
        timing = rollout.normalize_servo_timing(
            policy_hz=20.0,
            servo_hz=200.0,
            target_timeout_s=1.0,
            max_step_cm=1.0,
        )
        thread = rollout.InterpolatedServoThread(
            robot,
            initial_pose=np.zeros(6, dtype=float),
            timing=timing,
        )
        thread.start()
        step_m = 0.005
        try:
            for policy_step in range(3):
                thread.publish_segment(
                    np.array([
                        step_m * (policy_step + 1),
                        0.0, 0.0, 0.0, 0.0, 0.0,
                    ]),
                    policy_step=policy_step,
                )
                time.sleep(timing.policy_period_s)
            # Let the last committed segment finish before sampling the command.
            time.sleep(timing.policy_period_s)
        finally:
            thread.stop_and_join("test_complete")
        # Three committed steps of step_m each; the commanded frame must have
        # travelled essentially all of it, not a lagged fraction.
        commanded_x = robot.servo_calls[-1]["target_pose"][0]
        self.assertGreater(commanded_x, 2.5 * step_m)

    def test_servo_watchdog_stops_on_stale_target(self):
        robot = self.FakeRobotControl()
        timing = rollout.normalize_servo_timing(
            policy_hz=20.0,
            servo_hz=200.0,
            target_timeout_s=0.02,
            max_step_cm=1.0,
        )
        thread = rollout.InterpolatedServoThread(
            robot,
            initial_pose=np.zeros(6, dtype=float),
            timing=timing,
        )
        thread.start()
        # The watchdog only arms once the policy loop has committed something,
        # so a stale-target stop requires a publish first.
        thread.publish_segment(
            np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
            policy_step=0,
        )
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        with self.assertRaises(rollout.ServoLoopSafetyStop):
            thread.raise_if_failed()
        snapshot = thread.snapshot()
        self.assertTrue(snapshot.watchdog_stopped)
        self.assertIn("stale_target", snapshot.stop_reason)
        self.assertEqual(robot.stop_count, 1)

    def test_servo_watchdog_grants_startup_grace_before_first_publish(self):
        # Cold-start costs (first torch forward pass, video writer creation) can
        # exceed the steady-state target timeout. Killing the run before step 0
        # ever published would make those costs look like a hardware fault.
        robot = self.FakeRobotControl()
        timing = rollout.normalize_servo_timing(
            policy_hz=20.0,
            servo_hz=200.0,
            target_timeout_s=0.02,
            max_step_cm=1.0,
        )
        self.assertGreater(timing.startup_grace_s, 0.2)
        thread = rollout.InterpolatedServoThread(
            robot,
            initial_pose=np.zeros(6, dtype=float),
            timing=timing,
        )
        thread.start()
        time.sleep(0.15)
        still_running = thread.is_alive()
        thread.stop_and_join("test_complete")
        self.assertTrue(still_running)
        self.assertFalse(thread.snapshot().watchdog_stopped)

    def test_resync_collapses_committed_frame_onto_measured_pose(self):
        robot = self.FakeRobotControl()
        timing = rollout.normalize_servo_timing(
            policy_hz=20.0,
            servo_hz=200.0,
            target_timeout_s=1.0,
            max_step_cm=1.0,
        )
        thread = rollout.InterpolatedServoThread(
            robot,
            initial_pose=np.zeros(6, dtype=float),
            timing=timing,
        )
        runaway = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
        thread.publish_segment(runaway, policy_step=0)
        thread.publish_segment(runaway * 2.0, policy_step=1)
        measured = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0])
        thread.resync_committed_frame(measured)
        self.assertEqual(len(thread._segments), 1)
        np.testing.assert_allclose(thread.committed_end_pose(), measured)
        # The next committed step must start from the resynced pose, not from the
        # abandoned runaway endpoint.
        thread.publish_segment(
            np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0]),
            policy_step=2,
        )
        np.testing.assert_allclose(list(thread._segments)[-1].start_pose, measured)

    def test_servo_returning_false_propagates_failure(self):
        robot = self.FakeRobotControl(servo_return=False)
        timing = rollout.normalize_servo_timing(
            policy_hz=20.0,
            servo_hz=200.0,
            target_timeout_s=1.0,
        )
        thread = rollout.InterpolatedServoThread(
            robot,
            initial_pose=np.zeros(6, dtype=float),
            timing=timing,
        )
        thread.start()
        thread.join(timeout=0.2)
        self.assertFalse(thread.is_alive())
        with self.assertRaisesRegex(
            rollout.ServoLoopFailure,
            "servoL returned failure",
        ):
            thread.raise_if_failed()
        self.assertEqual(robot.stop_count, 1)


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

    def test_summary_contains_policy_and_servo_diagnostics(self):
        logger_path = (
            DEPLOYMENT_SRC / "callbacks" / "deployment_rollout_logger.py"
        )
        spec = importlib.util.spec_from_file_location(
            "deployment_rollout_logger_servo_test",
            logger_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as out_root:
            logger = module.DeploymentRolloutLogger(
                out_root=out_root,
                metadata={"duration_target_s": 1.0},
            )
            for step in range(2):
                logger.record_step({
                    "step": step,
                    "t_task_s": 0.05 * step,
                    "in_zpd": True,
                    "control_loop_hz_inst": 20.0,
                    "policy_loop_hz_inst": 20.0,
                    "policy_deadline_overrun_s": 0.002,
                    "policy_deadline_overrun": True,
                    "servo_loop_hz_inst": 125.0,
                    "servo_target_age_s": 0.01,
                    "servo_tracking_error_cm": 0.2,
                    "servo_target_error_cm": 0.3,
                    "servo_deadline_overrun_s": 0.001,
                    "servo_deadline_overrun_count": step + 1,
                    "servo_watchdog_stopped": False,
                })
            summary = logger.close(done_reason="timeout")
            self.assertAlmostEqual(summary["policy_loop_rate_hz_mean"], 20.0)
            self.assertAlmostEqual(summary["servo_loop_rate_hz_mean"], 125.0)
            self.assertAlmostEqual(summary["servo_tracking_error_cm_mean"], 0.2)
            self.assertEqual(summary["policy_deadline_overrun_count"], 2)
            self.assertEqual(summary["servo_deadline_overrun_count"], 2.0)
            header = (logger.rollout_dir / "timeseries.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertIn("servo_loop_hz_inst", header)
            self.assertIn("servo_tracking_error_cm", header)


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


class MicrorobotVisionModeTests(unittest.TestCase):
    @staticmethod
    def make_args(**overrides):
        values = {
            "microrobot_vision": "auto",
            "hand_source": "virtual",
            "no_display": True,
            "save_video": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_auto_disables_yolo_for_every_virtual_hand_mode(self):
        cases = [
            {},
            {"no_display": False},
            {"save_video": True},
            {"no_display": False, "save_video": True},
        ]
        for overrides in cases:
            with self.subTest(**overrides):
                mode, reason = rollout.resolve_microrobot_vision_mode(
                    self.make_args(**overrides)
                )
                self.assertEqual(mode, "none")
                self.assertEqual(reason, "virtual_hand_uses_ur_rtde_tcp")

    def test_auto_keeps_yolo_for_camera_hand(self):
        mode, reason = rollout.resolve_microrobot_vision_mode(
            self.make_args(hand_source="camera")
        )
        self.assertEqual(mode, "yolo")
        self.assertIsNone(reason)

    def test_explicit_modes_override_auto(self):
        for requested in ("yolo", "none"):
            with self.subTest(requested=requested):
                mode, reason = rollout.resolve_microrobot_vision_mode(
                    self.make_args(microrobot_vision=requested)
                )
                self.assertEqual(mode, requested)
                self.assertIsNone(reason)

    def test_disabled_mode_does_not_require_or_construct_yolo(self):
        missing = Path("C:/definitely_missing/best.onnx")
        self.assertIsNone(
            rollout.resolve_vision_model_path(missing, yolo_enabled=False)
        )

        class FailingYolo:
            def __init__(self, _):
                raise AssertionError("YOLO should not be constructed")

        self.assertIsNone(
            rollout.load_microrobot_model(
                missing,
                yolo_enabled=False,
                yolo_class=FailingYolo,
            )
        )

    def test_enabled_mode_still_requires_and_constructs_yolo(self):
        missing = Path("C:/definitely_missing/best.onnx")
        with self.assertRaises(FileNotFoundError):
            rollout.resolve_vision_model_path(missing, yolo_enabled=True)

        constructed = []

        class FakeYolo:
            def __init__(self, path):
                constructed.append(path)

        model = rollout.load_microrobot_model(
            Path("C:/models/best.onnx"),
            yolo_enabled=True,
            yolo_class=FakeYolo,
        )
        self.assertIsInstance(model, FakeYolo)
        self.assertEqual(constructed, ["C:\\models\\best.onnx"])


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
            policy_hz=20.0,
            servo_mode="interpolated",
            servo_hz=125.0,
            servo_target_timeout_s=0.15,
            inter_run_delay=3.0,
            save_video=False,
            no_display=True,
            microrobot_vision="auto",
        )

    def test_forty_matched_configurations_generate_eighty_runs(self):
        manifest = batch.build_manifest(
            self.make_args(),
            Path("C:/temporary/deployment_batch_test"),
        )
        self.assertEqual(manifest["schema_version"], 5)
        self.assertEqual(
            manifest["parameters"]["microrobot_vision_mode_requested"],
            "auto",
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
            self.assertIn("--control-hz", run["command"])
            self.assertIn("--servo-mode", run["command"])
            self.assertIn("--servo-hz", run["command"])
            self.assertNotIn("--microrobot-vision", run["command"])
            self.assertEqual(run["policy_freq_target_hz"], 20.0)
            self.assertEqual(run["servo_freq_target_hz"], 125.0)

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

    def test_explicit_microrobot_vision_override_is_passed_to_children(self):
        args = self.make_args()
        args.microrobot_vision = "none"
        manifest = batch.build_manifest(
            args,
            Path("C:/temporary/deployment_batch_test"),
        )
        self.assertEqual(
            manifest["parameters"]["microrobot_vision_mode_requested"],
            "none",
        )
        for run in manifest["runs"]:
            option_index = run["command"].index("--microrobot-vision")
            self.assertEqual(run["command"][option_index + 1], "none")

    def test_legacy_manifest_resume_forces_legacy_servo_mode(self):
        legacy = {
            "schema_version": 4,
            "parameters": {},
            "runs": [{
                "run_id": "trial001_league",
                "command": ["python", "record_deployment_chase.py"],
            }],
        }
        normalized = batch.normalize_manifest(legacy)
        run = normalized["runs"][0]
        self.assertEqual(run["servo_mode"], "legacy")
        self.assertFalse(run["timing_settings_required"])
        self.assertIn("--control-hz", run["command"])
        option_index = run["command"].index("--servo-mode")
        self.assertEqual(run["command"][option_index + 1], "legacy")
        backfilled = batch.backfill_result_timing(
            {"control_loop_rate_hz_mean": "16.4"},
            run,
        )
        self.assertEqual(backfilled["servo_mode"], "legacy")
        self.assertEqual(backfilled["policy_freq_target_hz"], 20.0)
        self.assertEqual(backfilled["policy_loop_rate_hz_mean"], "16.4")

    def test_interpolated_batch_rejects_servo_slower_than_policy(self):
        args = self.make_args()
        args.servo_hz = 10.0
        with self.assertRaises(ValueError):
            batch.build_manifest(
                args,
                Path("C:/temporary/deployment_batch_test"),
            )

    def test_disabled_yolo_metrics_are_unavailable_not_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir)
            (rollout_dir / "metadata.json").write_text(
                json.dumps({
                    "zpd_low_cm": 3.5,
                    "zpd_high_cm": 5.5,
                    "duration_target_s": 60.0,
                    "virtual_hand_stride_cm": 0.2,
                    "virtual_hand_smoothing_alpha": 0.7,
                    "virtual_hand_delay_frames": 1,
                    "microrobot_yolo_enabled": False,
                    "workspace_width_cm": 15.0,
                    "workspace_height_cm": 10.0,
                    "workspace_margin_cm": 0.3,
                }),
                encoding="utf-8",
            )
            (rollout_dir / "summary.json").write_text(
                json.dumps({
                    "done_reason": "caught",
                    "duration_s": 1.0,
                    "duration_target_s": 60.0,
                    "num_control_steps": 2,
                    "zpd_time_s": 1.0,
                    "tiz_fixed_horizon_fraction": 1.0 / 60.0,
                    "zpd_observed_occupancy_fraction": 1.0,
                    "zpd_occupancy_fraction": 1.0,
                    "safety_stop_count": 0,
                }),
                encoding="utf-8",
            )
            (rollout_dir / "timeseries.csv").write_text(
                "t_task_s,distance_cm,in_zpd,robot_x_cm,robot_y_cm,"
                "microrobot_detected,microrobot_x_cm,microrobot_y_cm,"
                "target_clipped,target_step_limited\n"
                "0.0,4.5,true,7.5,5.0,false,,,false,false\n"
                "1.0,4.0,true,7.6,5.0,false,,,false,false\n",
                encoding="utf-8",
            )
            run = {
                "run_id": "trial001_league",
                "pair_id": "trial001",
                "trial_index": 1,
                "internal_seed": 1,
                "order_in_pair": 1,
                "controller": "league",
                "duration_target_s": 60.0,
            }
            metrics = batch.extract_metrics(run, rollout_dir)
            self.assertIsNone(metrics["microrobot_detection_fraction"])
            self.assertIsNone(metrics["microrobot_tcp_error_cm_mean"])
            self.assertIsNone(metrics["microrobot_tcp_error_cm_p95"])

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
            "policy_freq_target_hz": run["policy_freq_target_hz"],
            "servo_mode": run["servo_mode"],
            "servo_freq_target_hz": run["servo_freq_target_hz"],
            "servo_target_timeout_s": run["servo_target_timeout_s"],
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

        wrong_servo = dict(row)
        wrong_servo["servo_freq_target_hz"] = 100.0
        self.assertFalse(batch.result_matches_run(wrong_servo, run))


if __name__ == "__main__":
    unittest.main()
