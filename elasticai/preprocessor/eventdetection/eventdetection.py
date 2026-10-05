from dataclasses import dataclass
from enum import Enum
from logging import Logger, getLogger
from pathlib import Path

import numpy as np

from elasticai.creator_plugins.eventdetection.src import c_compile
from elasticai.preprocessor._common_func import CommonDigitalFunctions


class EventDetectionTargets(Enum):
    Normal = "normal"
    PosHyst = "pos_hyst"
    NegHyst = "neg_hyst"
    DoubleHyst = "double_hyst"


@dataclass
class EventDetectionSettings:
    """Settings class for configuring the properties of the event detection module
    Attributes:
        method:           Applied types of hysteresis [
            'normal':      event_on/off -> threshold,
            'pos_hyst':    event_on -> thr + h, event_off -> thr,
            'neg_hyst':    event_on -> thr, event_off thr - h,
            'double_hyst': event_on -> thr + h, event_off -> thr - h]
        window_size:    Hysteresis window
        out_invert:     Is event low [True] or event high [False]
    """

    method: EventDetectionTargets
    window_size: int
    out_invert: bool


DefaultEventDetectionSettings = EventDetectionSettings(
    method=EventDetectionTargets.Normal,
    window_size=10,
    out_invert=False,
)


class EventDetection:
    _settings: EventDetectionSettings
    _int_state: bool

    def __init__(self, settings: EventDetectionSettings) -> None:
        """Class for detecting events in a transient input signal
        :param settings:    Class SettingsEventDetection for configuring the properties
        :return:            None
        """
        self._logger: Logger = getLogger(__name__)
        self._settings = settings
        if isinstance(settings.method, str):
            self._settings.method = EventDetectionTargets(settings.method)

    def _type_hysteresis(self, threshold: int) -> list:
        thr_zero = threshold
        thr_pos = thr_zero + self._settings.window_size
        thr_neg = thr_zero - self._settings.window_size

        match self._settings.method:
            case EventDetectionTargets.Normal:
                list_out = [thr_zero, thr_zero]
            case EventDetectionTargets.PosHyst:
                list_out = [thr_pos, thr_zero]
            case EventDetectionTargets.NegHyst:
                list_out = [thr_zero, thr_neg]
            case EventDetectionTargets.DoubleHyst:
                list_out = [thr_pos, thr_neg]
            case _:
                raise NotImplementedError(f"Hysteresis_method '{self._settings.method}' does not exist.")
        return list_out

    def get_events(self, xin: np.ndarray, threshold: np.ndarray) -> np.ndarray:
        """Extracting the event decision in stream output based on the input and threshold value
        :param xin:         Numpy array with input signal
        :param threshold:   Numpy array with threshold values
        :return:            Numpy array with event decision
        """
        xout = np.zeros(len(xin), dtype=bool)
        self._int_state = False

        for idx, (val, thr) in enumerate(zip(xin, threshold)):
            thr = self._type_hysteresis(thr)
            if self._int_state:
                xout[idx] = val >= thr[1]
            else:
                xout[idx] = val >= thr[0]
            self._int_state = xout[idx]
        return ~xout if self._settings.out_invert else xout

    def get_events_position(self, xin: np.ndarray, threshold: np.ndarray) -> list:
        """Extracting the event decision in stream output based on the input and threshold value
        :param xin:         Numpy array with input signal
        :param threshold:   Numpy array with threshold values
        :return:            List with timestamps of edge decision
        """
        func = CommonDigitalFunctions()

        xtrg = self.get_events(xin=xin, threshold=threshold)
        if not self._settings.out_invert:
            return func._extract_rising_edge(xtrg)
        else:
            return func._extract_falling_edge(xtrg)

    def create_design(
        self,
        target: str,
        bitwidth: int,
        id: str,
        path2save: Path,
        signed: bool = True,
    ) -> None:
        """
        Generate the hardware design to detect events on hardware
        :param target:          Target platform [
                                    "mcu",
                                    "pc",
                                    "fpga",
                                    "asic"],
        :param bitwidth:        Bitwidth,
        :param id:              ID of the target structure,
        :param path2save:       Path to save event detection design files
        :param signed:          for use in datatype,
        :return:                None,
        """

        supported_targets = ["mcu", "pc", "fpga", "asic"]
        if target.lower() not in supported_targets:
            raise ValueError(f"Target {target} is not supported: only {supported_targets}")
        assert bitwidth in range(2, 33), "Bitwidth must be between 2 and 32"

        if target.lower() in ["mcu", "pc"]:
            self._create_design_c(
                id=id,
                bitwidth=bitwidth,
                signed=signed,
                path2save=path2save,
            )
        else:
            self._create_design_fpga(
                id=id,
                bitwidth=bitwidth,
                signed=signed,
                path2save=path2save,
            )

    def _create_design_c(
        self,
        id: str,
        bitwidth: int,
        signed: bool,
        path2save: Path,
    ) -> None:

        c_compile.build_eventdetection(
            hysteresis=self._settings.window_size,
            hysteresis_type=self._settings.method.value,
            out_invert=self._settings.out_invert,
            bitwidth=bitwidth,
            signed=signed,
            path2save=path2save,
            eventdetection_id=id,
            define_path=".",
        )

    def _create_design_fpga(
        self,
        id: str,
        bitwidth: int,
        signed: bool,
        path2save: Path,
    ) -> None:
        from elasticai.creator_plugins.eventdetection.utils import load_and_plugin

        module_appendix = "un" if not signed else ""
        thr_on, thr_off = self._type_hysteresis(0)
        match self._settings.method:
            case EventDetectionTargets.Normal:
                load_and_plugin(
                    type=f"eventdetector_sub_{module_appendix}signed",
                    id=id,
                    params={"BITWIDTH": bitwidth, "OUT_INVERT": int(self._settings.out_invert)},
                    packages=["eventdetection"],
                    path2save=path2save,
                )
            case _:
                load_and_plugin(
                    type=f"eventdetector_hyst_{module_appendix}signed",
                    id=id,
                    params={
                        "BITWIDTH": bitwidth,
                        "THR_ON": abs(thr_on),
                        "THR_OFF": abs(thr_off),
                        "OUT_INVERT": int(self._settings.out_invert),
                    },
                    packages=["eventdetection"],
                    path2save=path2save,
                )
