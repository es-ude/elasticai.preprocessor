from os import makedirs
from os.path import join, exists
from unittest import TestCase, main
from denspp.offline.preprocessing import SettingsFilter

from denspp.translate import get_path_to_project_start
from denspp.translate.c_memory.waveform_lut import build_waveform_lut


settings = SettingsFilter(
    gain=1,
    fs=2e3,
    n_order=51,
    f_filt=[100],
    type="fir",
    f_type="butter",
    b_type="lowpass",
)


class TestCWaveformLUT(TestCase):
    path2save = join(get_path_to_project_start(), "build_temp", "memory_c")
    makedirs(path2save, exist_ok=True)
    bitwidth = 16

    def test_build_waveform_sine_full(self):
        build_waveform_lut(
            bitwidth=self.bitwidth,
            f_rpt=250e3,
            f_wvf=10e3,
            signed=True,
            do_optimized=False,
            type_sig="sine",
            module_id="0",
            path2save=self.path2save,
        )
        chck_files = [
            "waveform_lut_template.h",
            "waveform_lut_sine0.h",
            "waveform_lut_sine0.c",
        ]
        chck = [exists(join(self.path2save, file)) for file in chck_files]
        self.assertTrue(all(chck))

    def test_build_waveform_sine_opt(self):
        build_waveform_lut(
            bitwidth=self.bitwidth,
            f_rpt=250e3,
            f_wvf=10e3,
            signed=True,
            do_optimized=True,
            type_sig="sine",
            module_id="1",
            path2save=self.path2save,
        )
        chck_files = [
            "waveform_lut_template.h",
            "waveform_lut_sine1.h",
            "waveform_lut_sine1.c",
        ]
        chck = [exists(join(self.path2save, file)) for file in chck_files]
        self.assertTrue(all(chck))

    def test_build_waveform_tri_full(self):
        build_waveform_lut(
            bitwidth=self.bitwidth,
            f_rpt=250e3,
            f_wvf=10e3,
            signed=True,
            do_optimized=False,
            type_sig="tri",
            module_id="0",
            path2save=self.path2save,
        )
        chck_files = [
            "waveform_lut_template.h",
            "waveform_lut_tri0.h",
            "waveform_lut_tri0.c",
        ]
        chck = [exists(join(self.path2save, file)) for file in chck_files]
        self.assertTrue(all(chck))

    def test_build_waveform_rect_full(self):
        build_waveform_lut(
            bitwidth=self.bitwidth,
            f_rpt=250e3,
            f_wvf=10e3,
            signed=True,
            do_optimized=False,
            type_sig="rect",
            module_id="0",
            path2save=self.path2save,
        )
        chck_files = [
            "waveform_lut_template.h",
            "waveform_lut_rect0.h",
            "waveform_lut_rect0.c",
        ]
        chck = [exists(join(self.path2save, file)) for file in chck_files]
        self.assertTrue(all(chck))


if __name__ == "__main__":
    main()
