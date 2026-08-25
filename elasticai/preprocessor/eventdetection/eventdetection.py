from dataclasses import dataclass
from enum import IntEnum
from logging import Logger, getLogger
from pathlib import Path

import numpy as np

from elasticai.creator_plugins.eventdetection.src import c_compile


class TargetsEventDetection(IntEnum):
    Normal = 0
    PosHyst = 1
    NegHyst = 2
    DoubleHyst = 3


@dataclass
class SettingsEventDetection:
    """Settings class for configuring the properties of the eventdetection module
    Attributes:
        hysteresis: Hysteresis window
        hysteresis_type: Applied types of hysteresis [
            'normal':      event_on/off -> threshold,
            'pos_hyst':    event_on -> thr + h, event_off -> thr,
            'neg_hyst':    event_on -> thr, event_off thr - h,
            'double_hyst': event_on -> thr + h, event_off -> thr - h]
        out_invert: Is event low [True] or event high [False]
    """

    hysteresis:      int 
    hysteresis_type: int | TargetsEventDetection
    out_invert:      bool

DefaultSettingsEventDetection = SettingsEventDetection(
    hysteresis=10,
    hysteresis_type=TargetsEventDetection.Normal,
    out_invert=False,
)


class EventDetection:
    _settings: SettingsEventDetection
    _int_state: bool

    def __init__(self, settings: SettingsEventDetection) -> None:
        """Class for detecting events in a transient input signal
        :param settings:    Class SettingsEventDetection for configuring the properties
        :return:            None
        """
        self._logger: Logger = getLogger(__name__)
        self._settings = settings
        self._cmap = {
            TargetsEventDetection.Normal: "eventdetection_normal",
            TargetsEventDetection.PosHyst: "eventdetection_pos_hyst",
            TargetsEventDetection.NegHyst: "eventdetection_neg_hyst",
            TargetsEventDetection.DoubleHyst: "eventdetection_double_hyst",
        }

    def _map_type_to_c(self) -> str:
        if self._settings.hysteresis_type not in list(self._cmap.keys()):
            raise ValueError(
                f"Hysteresis-Type '{self._settings.hysteresis_type}' in C generation is not supported."
            )
        return self._cmap[self._settings.hysteresis_type]

    def _type_hysteresis(self, method: int | TargetsEventDetection, threshold: int) -> list:
        thr_zero = threshold
        thr_pos = thr_zero + self._settings.hysteresis
        thr_neg = thr_zero - self._settings.hysteresis
        match method:
            case TargetsEventDetection.Normal:
                list_out = [thr_zero, thr_zero]
            case TargetsEventDetection.PosHyst:
                if self._settings.out_invert:
                    list_out = [thr_zero, thr_neg]
                else:
                    list_out = [thr_pos, thr_zero]
            case TargetsEventDetection.NegHyst:
                if self._settings.out_invert:
                    list_out = [thr_pos, thr_zero]
                else:
                    list_out = [thr_zero, thr_neg]
            case TargetsEventDetection.DoubleHyst:
                if self._settings.out_invert:
                    list_out = [thr_neg, thr_pos]
                else:
                    list_out = [thr_pos, thr_neg]
            case _:
                raise NotImplementedError(
                    f"Hysteresis_type '{self._settings.hysteresis_type}' does not exist."
                )
        return list_out

    def get_events(self, xin: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        xout = np.zeros(len(xin), dtype=bool)
        self._int_state = False
        # int_state == 0 -> no event, int_state == 1 -> event detected
        for idx, val in enumerate(xin):
            thr = self._type_hysteresis(self._settings.hysteresis_type, thresholds[idx].astype(int))
            if self._settings.out_invert:
                if self._int_state:
                    xout[idx] = val <= thr[1]
                else:
                    xout[idx] = val <= thr[0]
                self._int_state = xout[idx]
            else:
                if self._int_state:
                    xout[idx] = val >= thr[1]
                else:
                    xout[idx] = val >= thr[0]
                self._int_state = xout[idx]
        return xout

    def get_events_position(self, xout: np.ndarray) -> np.ndarray:
        ev_pos = np.zeros(len(xout), dtype=bool)
        ev_pos[0] = xout[0]
        ev_pos[1:] = xout[1:0] & ~xout[:-1]
        return ev_pos

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
        :param path2save:       Path to save eventdetection design files
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
            self._create_design_fpga()

    def _create_design_c(
        self,
        id: str,
        bitwidth: int,
        signed: bool,
        path2save: Path,
    ) -> None:
        hysteresis_c_type = self._map_type_to_c()
        c_compile.build_eventdetection(
            hysteresis=self._settings.hysteresis,
            hysteresis_type=hysteresis_c_type,
            bitwidth=bitwidth,
            signed=signed,
            path2save=path2save,
            eventdetection_id=id,
            define_path=".",
        )

    def _create_design_fpga(self) -> None:
        raise NotImplementedError("FPGAs are not yet supported")
