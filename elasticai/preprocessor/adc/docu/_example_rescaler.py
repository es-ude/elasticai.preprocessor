from pathlib import Path

import numpy as np

from elasticai.preprocessor import PreprocessingSequential, get_path_to_project
from elasticai.preprocessor._data_plot import Data, load_smatable_data, plot_data
from elasticai.preprocessor.adc import Rescaler, RescalerSettings, RescalerTargets
from elasticai.preprocessor.sequential import CreateSequentialSettings


def process_data(data: Data) -> Data:
    sets = RescalerSettings(method=RescalerTargets.FxP, total_bits=8, frac_bits=7, is_signed=True)

    dut = PreprocessingSequential(
        Rescaler(settings=sets),
    )
    out = dut(
        x=np.asarray(1e4 * data.data, dtype=int),
        fs=data.sampling_rate,
    )
    dut.create_design(
        settings=CreateSequentialSettings(
            target="mcu",
            total_bitwidth=sets.total_bits,
            frac_bitwidth=sets.frac_bits,
            do_signed=sets.is_signed,
            path2build=get_path_to_project("build"),
        )
    )
    return Data(
        sampling_rate=out.sampling_rate,
        time=np.arange(start=0, stop=out.data.shape[-1], step=1) / out.sampling_rate,
        data=out.data,
        channel_id=data.channel_id,
    )


if __name__ == "__main__":
    data_orig = load_smatable_data(
        path2data=get_path_to_project()
        / "smatable-data"
        / "raw-data"
        / "sub-P001"
        / "ses-S001/knock/sub-P001_ses-S001_task-Default_run-001_knock.xdf"
    )
    data_proc = process_data(data_orig)
    plot_data(
        data_orig=data_orig,
        data_filt=data_proc,
        path2save=Path(__file__).parent.absolute() / "example_rescaler.svg",
        yscale=1e0,
        ylabel_out="out [FxP(8, 7)]",
    )
