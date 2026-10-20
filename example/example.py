import numpy as np

from elasticai.preprocessor._data_plot import Data, load_smatable_data, plot_data
from elasticai.preprocessor import get_path_to_project, PreprocessingSequential
from elasticai.preprocessor.sequential import CreateSequentialSettings
from elasticai.preprocessor.downsampling import DownSamplingSettings, DownSampling, DownSamplingTargets


def process_data(data: Data) -> Data:
    sets_down = DownSamplingSettings(
        method=DownSamplingTargets.Polyphase,
        num_stages=1,
        sampling_rate=data.sampling_rate,
        dsr=2
    )

    dut = PreprocessingSequential(
        DownSampling(settings=sets_down),
    )
    out = dut(
        x=data.data,
        fs=data.sampling_rate,
    )
    dut.create_design(
        settings=CreateSequentialSettings(
            target="fpga",
            total_bitwidth=8,
            frac_bitwidth=5,
            do_signed=True,
            path2build=get_path_to_project("build")
        )
    )

    return Data(
        sampling_rate=out.sampling_rate,
        time=np.arange(start=0, stop=out.data.shape[-1], step=1) / out.sampling_rate,
        data=out.data,
        channel_id=data.channel_id
    )


if __name__ == "__main__":
    data_orig = load_smatable_data(
        path2data=get_path_to_project() / "smatable-data" / "raw-data" / "sub-P001" / "ses-S001/knock/sub-P001_ses-S001_task-Default_run-001_knock.xdf"
    )
    data_proc = process_data(data_orig)
    plot_data(data_orig=data_orig, data_filt=data_proc)
