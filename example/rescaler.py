import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from elasticai.preprocessor._data_plot import Data, load_smatable_data, plot_data
from elasticai.preprocessor import get_path_to_project, PreprocessingSequential, DataTransient
from elasticai.preprocessor.sequential import CreateSequentialSettings
from elasticai.preprocessor.adc import Rescaler, RescalerTargets, RescalerSettings


def plot_hist_flat(
    data_orig: Data,
    data_filt: Data,
    bins: int = 201,
    path2save: Path | None = None,
) -> None:
    def _stats_text(x: np.ndarray) -> str:
        return (
            f"mean   = {np.mean(x):.3f}\n"
            f"std    = {np.std(x):.3f}\n"
            f"median = {np.median(x):.3f}\n"
            f"min    = {np.min(x):.3f}\n"
            f"max    = {np.max(x):.3f}"
        )

    flat_orig = np.asarray(data_orig.data).ravel()
    flat_filt = np.asarray(data_filt.data).ravel()

    fig, axs = plt.subplots(1, 2, sharey=False, figsize=(10, 4))

    for ax, x, color, title in (
        (axs[0], flat_orig, "k", "Original"),
        (axs[1], flat_filt, "r", "Rescaled"),
    ):
        ax.hist(x, bins=bins, color=color)
        ax.set_title(title)
        ax.set_xlabel(f"Amplitude")
        ax.set_ylabel("Count")
        ax.grid(True)
        ax.text(
            0.97, 0.97, _stats_text(x),
            transform=ax.transAxes,
            ha="right", va="top",
            family="monospace", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    fig.tight_layout()

    if path2save is not None:
        fig.savefig(path2save.with_stem(path2save.stem + "_hist"))
        plt.close(fig)


def process_data(data: Data) -> Data:
    sets = RescalerSettings(
        method=RescalerTargets.IntAsym,
        total_bits=6,
        frac_bits=0,
        is_signed=True
    )

    dut = Rescaler(settings=sets)
    tdata = np.asarray(data.data * 10000, dtype=int)

    cdata = list()
    for hdata in tdata:
        out = dut(DataTransient(
            data=hdata,
            sampling_rate=data.sampling_rate
        ))
        cdata.append(out.data)
        print(dut._get_data_statistics(hdata))
        print(dut._scaling_factor, dut._zero_point)

    return Data(
        sampling_rate=out.sampling_rate,
        time=np.arange(start=0, stop=cdata[-1].shape[-1], step=1) / out.sampling_rate,
        data=np.asarray(cdata),
        channel_id=data.channel_id
    )


if __name__ == "__main__":
    data_orig = load_smatable_data(
        path2data=get_path_to_project() / "smatable-data" / "raw-data" / "sub-P001" / "ses-S001/knock/sub-P001_ses-S001_task-Default_run-001_knock.xdf"
    )
    data_orig.data += 0.05
    data_proc = process_data(data_orig)

    plot_hist_flat(data_orig=data_orig, data_filt=data_proc)
    plot_data(data_orig=data_orig, data_filt=data_proc, yscale=1, ylabel_out="Rescaled [x LSB]")

