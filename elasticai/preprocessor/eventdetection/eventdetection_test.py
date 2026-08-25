import numpy as np
import pytest

from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.translation.cocotb_tmp import temporary_directory

from .eventdetection import (
    DefaultSettingsEventDetection,
    EventDetection,
    SettingsEventDetection,
    TargetsEventDetection,
)


def make_detector(
    hysteresis: int = 10,
    hysteresis_type: TargetsEventDetection = TargetsEventDetection.Normal,
    out_invert: bool = False,
):
    return EventDetection(
        SettingsEventDetection(
            window_size=hysteresis,
            type=hysteresis_type,
            out_invert=out_invert,
        )
    )


HYSTERESIS_TYPE_CONFIGS = {
    pytest.param(TargetsEventDetection.Normal),
    pytest.param(TargetsEventDetection.PosHyst),
    pytest.param(TargetsEventDetection.NegHyst),
    pytest.param(TargetsEventDetection.DoubleHyst),
}

OUT_INVERT_CONFIGS = {
    pytest.param(True, id="inversion"),
    pytest.param(False, id="no_inversion"),
}


def make_test_triangle_array(lower_limit: int, upper_limit: int, thr: int):
    xin = np.concatenate(
        [np.arange(lower_limit, upper_limit + 1), np.arange(upper_limit - 1, lower_limit - 1, -1)]
    )
    thrs = np.full(len(xin), thr)
    return xin, thrs


class TestDefaultSettings:
    def test_hysteresis(self):
        assert DefaultSettingsEventDetection.window_size == 10

    def test_hysteresis_type(self):
        assert DefaultSettingsEventDetection.type == TargetsEventDetection.Normal

    def test_out_invert(self):
        assert not DefaultSettingsEventDetection.out_invert


class TestDetectEventOutputShape:
    def test_output_length_matches_input(self):
        det = make_detector()
        xin = np.array([10, 50, 150, 200, 80])
        thresholds = np.array([100, 100, 100, 100, 100])
        assert len(det.get_events(xin, thresholds)) == len(xin)

    def test_output_dtype_is_bool(self):
        det = make_detector()
        xout = det.get_events(np.array([50, 150]), np.array([100, 100]))
        assert xout.dtype == bool

    def test_empty_input_returns_empty(self):
        det = make_detector()
        xout = det.get_events(np.array([]), np.array([]))
        assert len(xout) == 0


class TestNormalMode:
    # hysteresis_type="normal" → thr_on=100, thr_off=100

    def test_above_threshold_is_event(self):
        det = make_detector(hysteresis_type=TargetsEventDetection.Normal)
        xout = det.get_events(np.array([100, 150, 200]), np.array([100, 100, 100]))
        np.testing.assert_array_equal(xout, [True, True, True])

    def test_below_threshold_is_no_event(self):
        det = make_detector(hysteresis_type=TargetsEventDetection.Normal)
        xout = det.get_events(np.array([0, 50, 99]), np.array([100, 100, 100]))
        np.testing.assert_array_equal(xout, [False, False, False])

    def test_switches_without_hysteresis(self):
        det = make_detector(hysteresis_type=TargetsEventDetection.Normal)
        xin = np.array([150, 50, 150, 50])
        thresholds = np.array([100, 100, 100, 100])
        xout = det.get_events(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False, True, False])

    def test_state_resets_at_start_of_each_call(self):
        det = make_detector(hysteresis_type=TargetsEventDetection.Normal)
        det.get_events(np.array([200]), np.array([100]))  # leaves _int_state = True
        xout = det.get_events(np.array([50]), np.array([100]))
        assert not xout[0]

    def test_switches_exact_at_thresholds_norm(self):
        det = make_detector(hysteresis_type=TargetsEventDetection.Normal)
        xin, thrs = make_test_triangle_array(95, 105, 100)
        xout = det.get_events(xin, thrs)
        np.testing.assert_array_equal(
            xout,
            np.concatenate(
                [
                    np.full(5, False, dtype=bool),
                    np.full(11, True, dtype=bool),
                    np.full(5, False, dtype=bool),
                ]
            ),
        )


class TestPosHystMode:
    # threshold=100, hysteresis=10
    # thr_on  = 100 + 10 = 110
    # thr_off = 100

    def test_does_not_trigger_below_thr_on(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.PosHyst)
        xin, thrs = make_test_triangle_array(95, 99, 100)
        xout = det.get_events(xin, thrs)
        np.testing.assert_array_equal(xout, np.full(9, False, dtype=bool))

    def test_triggers_at_thr_on(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.PosHyst)
        xout = det.get_events(np.array([110]), np.array([100]))
        assert xout[0]

    def test_stays_active_at_thr_off(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.PosHyst)
        xin = np.array([110, 105, 100])  # trigger at 130, stay at 110 and 100
        thresholds = np.array([100, 100, 100])
        xout = det.get_events(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, True, True])

    def test_deactivates_below_thr_off(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.PosHyst)
        xin = np.array([110, 99])
        thresholds = np.array([100, 100])
        xout = det.get_events(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False])

    def test_does_not_retrigger_below_thr_on(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.PosHyst)
        xin = np.array([110, 99, 109])  # deactivates, stays off below thr_on
        thresholds = np.array([100, 100, 100])
        xout = det.get_events(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False, False])

    def test_switches_exact_at_thresholds_pos(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.PosHyst)
        xin, thrs = make_test_triangle_array(95, 115, 100)
        xout = det.get_events(xin, thrs)
        np.testing.assert_array_equal(
            xout,
            np.concatenate(
                [
                    np.full(15, False, dtype=bool),
                    np.full(21, True, dtype=bool),
                    np.full(5, False, dtype=bool),
                ]
            ),
        )


class TestNegHystMode:
    # hysteresis=10
    # thr_on  = 100
    # thr_off = 100 - 10 = 90

    def test_triggers_at_thr_on(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.NegHyst)
        xout = det.get_events(np.array([100]), np.array([100]))
        assert xout[0]

    def test_stays_active_above_thr_off(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.NegHyst)
        xin = np.array([100, 95, 90])
        thresholds = np.array([100, 100, 100])
        xout = det.get_events(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, True, True])

    def test_deactivates_below_thr_off(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.NegHyst)
        xin = np.array([100, 89])
        thresholds = np.array([100, 100])
        xout = det.get_events(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False])

    def test_retriggers_at_thr_on(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.NegHyst)
        xin = np.array([100, 89, 100])  # on, off, on again
        thresholds = np.array([100, 100, 100])
        xout = det.get_events(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False, True])

    def test_switches_exact_at_thresholds_neg(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.NegHyst)
        xin, thrs = make_test_triangle_array(85, 105, 100)
        xout = det.get_events(xin, thrs)
        np.testing.assert_array_equal(
            xout,
            np.concatenate(
                [
                    np.full(15, False, dtype=bool),
                    np.full(21, True, dtype=bool),
                    np.full(5, False, dtype=bool),
                ]
            ),
        )


class TestDoubleHystMode:
    # hysteresis=10
    # thr_on  = 100 + 10 = 110
    # thr_off = 100 - 10 = 90

    def test_does_not_trigger_below_thr_on(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.DoubleHyst)
        xin = np.array([100, 105, 109])
        thresholds = np.array([100, 100, 100])
        xout = det.get_events(xin, thresholds)
        np.testing.assert_array_equal(xout, [False, False, False])

    def test_triggers_at_thr_on(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.DoubleHyst)
        xout = det.get_events(np.array([110]), np.array([100]))
        assert xout[0]

    def test_stays_active_above_thr_off(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.DoubleHyst)
        xin = np.array([110, 100, 90])
        thresholds = np.array([100, 100, 100])
        xout = det.get_events(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, True, True])

    def test_deactivates_below_thr_off(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.DoubleHyst)
        xin = np.array([110, 89])
        thresholds = np.array([100, 100])
        xout = det.get_events(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False])

    def test_does_not_retrigger_between_thresholds(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.DoubleHyst)
        xin = np.array([110, 89, 105])  # on, off, 110 is between thr_off and thr_on → stays off
        thresholds = np.array([100, 100, 100])
        xout = det.get_events(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False, False])

    def test_retrigggers_above_thr_on(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.DoubleHyst)
        xin = np.array([110, 89, 110])
        thresholds = np.array([100, 100, 100])
        xout = det.get_events(xin, thresholds)
        np.testing.assert_array_equal(xout, [True, False, True])

    def test_switches_exact_at_thresholds_double(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.PosHyst)
        xin, thrs = make_test_triangle_array(85, 115, 100)
        xout = det.get_events(xin, thrs)
        np.testing.assert_array_equal(
            xout,
            np.concatenate(
                [
                    np.full(25, False, dtype=bool),
                    np.full(21, True, dtype=bool),
                    np.full(15, False, dtype=bool),
                ]
            ),
        )


class TestUnknownHysteresisType:
    def test_raises_not_implemented_error(self):
        det = make_detector(hysteresis_type="unknown")
        with pytest.raises(NotImplementedError):
            det.get_events(np.array([100]), np.array([100]))


class TestGetEventsPosition:
    def test_single_event_positive(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.Normal, out_invert=False)
        xin = np.array([100, 90, 100, 100, 100])
        thr = 99 + np.zeros_like(xin)

        xout = det.get_events(xin, thr)
        xtmp = det.get_events_position(xin, thr)
        assert all(xout == np.array([True, False, True, True, True]))
        assert xtmp == [2]

    def test_single_event_invert(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.Normal, out_invert=True)
        xin = np.array([100, 90, 100, 100, 100])
        thr = 99 + np.zeros_like(xin)

        xout = det.get_events(xin, thr)
        xtmp = det.get_events_position(xin, thr)
        assert all(xout == np.array([False, True, False, False, False]))
        assert xtmp == [2]

    def test_multiple_events_positive(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.Normal, out_invert=False)
        xin = np.array([100, 90, 100, 100, 100, 90, 100, 100, 90, 90, 100, 100])
        thr = 99 + np.zeros_like(xin)

        xout = det.get_events(xin, thr)
        xtmp = det.get_events_position(xin, thr)
        assert all(
            xout == np.array([True, False, True, True, True, False, True, True, False, False, True, True])
        )
        assert xtmp == [2, 6, 10]

    def test_multiple_events_invert(self):
        det = make_detector(hysteresis=10, hysteresis_type=TargetsEventDetection.Normal, out_invert=True)
        xin = np.array([100, 90, 100, 100, 100, 90, 100, 100, 90, 90, 100, 100])
        thr = 99 + np.zeros_like(xin)

        xout = det.get_events(xin, thr)
        xtmp = det.get_events_position(xin, thr)
        assert all(
            xout
            == np.array([False, True, False, False, False, True, False, False, True, True, False, False])
        )
        assert xtmp == [2, 6, 10]


class TestCreateDesign:
    @pytest.mark.parametrize("target", ["mcu", "pc"])
    @pytest.mark.parametrize("out_invert", OUT_INVERT_CONFIGS)
    @pytest.mark.parametrize("hysteresis_type", HYSTERESIS_TYPE_CONFIGS)
    def test_create_design_generates_eventdetection_c_files(
        self,
        target: str,
        out_invert: bool,
        hysteresis_type: TargetsEventDetection,
    ) -> None:
        eventdetector = EventDetection(
            SettingsEventDetection(
                window_size=10,
                type=hysteresis_type,
                out_invert=out_invert,
            )
        )

        backup = get_path_to_project("build_test") / f"{hysteresis_type}"
        with temporary_directory(backup) as tmpdir:
            eventdetector.create_design(
                target=target,
                bitwidth=8,
                id="0",
                path2save=tmpdir,
                signed=True,
            )
            assert (tmpdir / "eventdetection_0.c").exists()
            assert (tmpdir / "eventdetection_0.h").exists()
            assert (tmpdir / "eventdetection_template.h").exists()
