from dataclasses import dataclass
from logging import Logger, getLogger
from pathlib import Path

import numpy as np

from elasticai.creator_plugins.eventdetection.src import c_compile


@dataclass
class SettingsEventDetection:
    """Settings class for configuring the properties of the eventdetection module
    Attributes: 
        # threshold:  threshold defining an event 
        hysteresis: Hysteresis window [%]
        hysteresis_type: Applied types of hysteresis [
            'normal': no hysteresis, 
            'pos_hyst': xin > 0 + h, 
            'neg_hyst': xin < 0 - h, 
            'double_hyst': später einschalten, später ausschalten]
        out_invert: Is event low [True] or event high [False]
    """
    hysteresis:      float 
    hysteresis_type: str
    

DefaultSettingsEventDetection = SettingsEventDetection(
    hysteresis=0.25,
    hysteresis_type="normal",
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
        
    def _type_hysteresis(self, mode: str, threshold: int) -> list:
        thr_zero = threshold
        thr_pos = thr_zero + thr_zero * self._settings.hysteresis
        thr_neg = thr_zero - thr_zero * self._settings.hysteresis
        match mode: 
            case "normal":
                # --- normal event_detection
                list_out = [thr_zero, thr_zero]
            case "pos_hyst": 
                # --- single Side, positive hysteresis
                list_out = [thr_pos, thr_zero]
            case "neg_hyst":
                # --- single Side, negative hysteresis
                list_out = [thr_zero, thr_neg]
            case "double_hyst":
                # --- double Side, 
                list_out = [thr_pos, thr_neg]
            case _:
                raise NotImplementedError(
                    f"Hysteresis_type '{self._settings.hysteresis_type}' does not exist."
                )
        return list_out

    def detect_event(self, xin: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        xout = np.zeros(len(xin), dtype=bool)
        self._int_state = False
        # int_state == 0 -> no event, int_state == 1 -> event detected
        for idx, val in enumerate(xin):     
            thr = self._type_hysteresis(self._settings.hysteresis_type, thresholds[idx].astype(int))
            if self._int_state:
                xout[idx] = val >= thr[1]
            else:
                xout[idx] = val >= thr[0]
            self._int_state = xout[idx]
        return xout

    def create_design(
        self,
        target: str,
        bitwidth: int,
        id: str,
        path2save: Path,
        hysteresis: float,
        hysteresis_type: str,
        signed: bool = True,
    ) -> None:
        """
        Generate the hardware design to detect events on hardware
        :param target:          Target platform ["mcu", "pc", "fpga", "asic"],
        :param bitwidth:        Bitwidth,
        :param id:              ID of the target structure,
        :param path2save:       Path to save eventdetection,
        :param signed:          for use in datatype,
        :param hysteresis:      relative hysteresis factor,
        :param hysteresis_type: applied type of hysteresis["normal", "pos_hyst", "neg_hyst", "double_hyst"],
        :return:                None,
        """
       
        supported_targets = ["mcu", "pc"]
        if target.lower() not in supported_targets:
            raise ValueError(f"Target {target} is not supported: only {supported_targets}")
        assert bitwidth in range(2, 33), "Bitwidth must be between 2 and 32"

        self._create_design_c(
            id=id,
            bitwidth=bitwidth,
            signed=signed,
            path2save=path2save,
            hysteresis=hysteresis,
            hysteresis_type=hysteresis_type,
        )

    def _create_design_c(
        self,
        id: str,
        bitwidth: int,
        signed: bool,
        path2save: Path,
        hysteresis: float = 0.25,
        hysteresis_type: str = "eventdetection_double_hyst",
    ) -> None:
        c_compile.build_eventdetection(
            hysteresis=hysteresis,
            hysteresis_type=hysteresis_type,
            bitwidth=bitwidth,
            signed=signed,
            path2save=path2save,
            eventdetection_id=id,
            define_path=".",
        )