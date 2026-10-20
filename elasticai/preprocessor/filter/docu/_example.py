from pathlib import Path

import numpy as np

from elasticai.preprocessor import PreprocessingSequential, get_path_to_project
from elasticai.preprocessor._data_plot import Data, load_smatable_data, plot_data
from elasticai.preprocessor.filter import Filtering, FilterSettings, FilterTargets
from elasticai.preprocessor.sequential import CreateSequentialSettings


def process_data(data: Data) -> Data:
    sets = FilterSettings(
        method=FilterTargets.IIR,
        sampling_rate=data.sampling_rate,
        n_order=2,
        f_filt=[20.0, 400.0],
        f_type="butter",
        b_type="bandpass",
    )

    dut = PreprocessingSequential(
        Filtering(settings=sets),
    )
    out = dut(
        x=data.data,
        fs=data.sampling_rate,
    )
    dut.create_design(
        settings=CreateSequentialSettings(
            target="fpga",
            total_bitwidth=8,
            frac_bitwidth=7,
            do_signed=True,
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
        path2save=Path(__file__).parent.absolute() / "example.svg",
        yscale=1e3,
        ylabel_out="Filter out [mV]",
    )
