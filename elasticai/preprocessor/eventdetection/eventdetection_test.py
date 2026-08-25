import numpy as np
import pytest

from .eventdetection import (
    DefaultSettingsEventDetection,
    EventDetection,
    SettingsEventDetection,
    TargetsEventDetection,
)


def make_detector(hysteresis: int = 10, hysteresis_type: str = "normal", out_invert: bool = False):
    return EventDetection(
        SettingsEventDetection(
            hysteresis=hysteresis, hysteresis_type=hysteresis_type, out_invert=out_invert
        )
    )


class TestDefaultSettings:
    def test_hysteresis(self):
        assert DefaultSettingsEventDetection.hysteresis == 10

    def test_hysteresis_type(self):
        assert DefaultSettingsEventDetection.hysteresis_type == TargetsEventDetection.Normal


class TestDetectEventOutputShape:
    def test_output_length_matches_input(self):
        det = make_detector()
        xin = np.array([10, 50, 150, 200, 80])
        thresholds = np.array([100, 100, 100, 100, 100])
        assert len(det.detect_event(xin, thresholds)) == len(xin)

    def test_output_dtype_is_bool(self):
        det = make_detector()
        xout = det.detect_event(np.array([50, 150]), np.array([100, 100]))
        assert xout.dtype == bool

    def test_empty_input_returns_empty(self):
        det = make_detector()
        xout = det.detect_event(np.array([]), np.array([]))
        assert len(xout) == 0


class TestNormalMode:
    # hysteresis_type="normal" → thr_on=100, thr_off=100

    def test_above_threshold_is_event(self):
        det = make_detector(hysteresis_type="normal")
        xout = det.detect_event(np.array([100, 150, 200]), np.array([100, 100, 100]))
        np.testing.assert_array_equal(xout, [True, True, True])

    def test_below_threshold_is_no_event(self):
        det = make_detector(hysteresis_type="normal")
        xout = det.detect_event(np.array([0, 50, 99]), np.array([100, 100, 100]))
        np.testing.assert_array_equal(xout, [False, False, False])

    def test_switches_without_hysteresis(self):
        det = make_detector(hysteresis_type="normal")
        xin = np.array([150, 50, 150, 50])
        thresholds = np.array([100, 100, 100, 100])
        xout = det.detect_event(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False, True, False])

    def test_state_resets_at_start_of_each_call(self):
        det = make_detector(hysteresis_type="normal")
        det.detect_event(np.array([200]), np.array([100]))  # leaves _int_state = True
        xout = det.detect_event(np.array([50]), np.array([100]))
        assert not xout[0]


class TestPosHystMode:
    # threshold=100, hysteresis=0.25
    # thr_on  = 100 + 100*0.25 = 125
    # thr_off = 100

    def test_does_not_trigger_below_thr_on(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="pos_hyst")
        xin = np.array([110, 120, 124])  # all below thr_on=125
        thresholds = np.array([100, 100, 100])
        xout = det.detect_event(xin, thresholds)
        np.testing.assert_array_equal(xout, [False, False, False])

    def test_triggers_at_thr_on(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="pos_hyst")
        xout = det.detect_event(np.array([125]), np.array([100]))
        assert xout[0]

    def test_stays_active_at_thr_off(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="pos_hyst")
        xin = np.array([130, 110, 100])  # trigger at 130, stay at 110 and 100
        thresholds = np.array([100, 100, 100])
        xout = det.detect_event(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, True, True])

    def test_deactivates_below_thr_off(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="pos_hyst")
        xin = np.array([130, 99])
        thresholds = np.array([100, 100])
        xout = det.detect_event(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False])

    def test_does_not_retrigger_below_thr_on(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="pos_hyst")
        xin = np.array([130, 99, 110])  # deactivates, stays off below thr_on
        thresholds = np.array([100, 100, 100])
        xout = det.detect_event(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False, False])


class TestNegHystMode:
    # hysteresis=0.25
    # thr_on  = 100
    # thr_off = 100 - 100*0.25 = 75

    def test_triggers_at_thr_on(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="neg_hyst")
        xout = det.detect_event(np.array([100]), np.array([100]))
        assert xout[0]

    def test_stays_active_above_thr_off(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="neg_hyst")
        xin = np.array([100, 80, 75])
        thresholds = np.array([100, 100, 100])
        xout = det.detect_event(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, True, True])

    def test_deactivates_below_thr_off(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="neg_hyst")
        xin = np.array([100, 74])
        thresholds = np.array([100, 100])
        xout = det.detect_event(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False])

    def test_retriggers_at_thr_on(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="neg_hyst")
        xin = np.array([100, 74, 100])  # on, off, on again
        thresholds = np.array([100, 100, 100])
        xout = det.detect_event(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False, True])


class TestDoubleHystMode:
    # hysteresis=0.25
    # thr_on  = 100 + 100*0.25 = 125
    # thr_off = 100 - 100*0.25 = 75

    def test_does_not_trigger_below_thr_on(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="double_hyst")
        xin = np.array([110, 120, 124])
        thresholds = np.array([100, 100, 100])
        xout = det.detect_event(xin, thresholds)
        np.testing.assert_array_equal(xout, [False, False, False])

    def test_triggers_at_thr_on(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="double_hyst")
        xout = det.detect_event(np.array([125]), np.array([100]))
        assert xout[0]

    def test_stays_active_above_thr_off(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="double_hyst")
        xin = np.array([130, 80, 75])
        thresholds = np.array([100, 100, 100])
        xout = det.detect_event(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, True, True])

    def test_deactivates_below_thr_off(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="double_hyst")
        xin = np.array([130, 74])
        thresholds = np.array([100, 100])
        xout = det.detect_event(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False])

    def test_does_not_retrigger_between_thresholds(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="double_hyst")
        xin = np.array([130, 74, 110])  # on, off, 110 is between thr_off and thr_on → stays off
        thresholds = np.array([100, 100, 100])
        xout = det.detect_event(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False, False])

    def test_retrigggers_above_thr_on(self):
        det = make_detector(hysteresis=0.25, hysteresis_type="double_hyst")
        xin = np.array([130, 74, 130])
        thresholds = np.array([100, 100, 100])
        xout = det.detect_event(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False, True])


class TestUnknownHysteresisType:
    def test_raises_not_implemented_error(self):
        det = make_detector(hysteresis_type="invalid")
        with pytest.raises(NotImplementedError):
            det.detect_event(np.array([100]), np.array([100]))
