import numpy as np

from copy import deepcopy
from pathlib import Path
from matplotlib import pyplot as plt

from elasticai.preprocessor._data_plot import Data, load_smatable_data, plot_data
from elasticai.preprocessor import get_path_to_project, DataTransient
from elasticai.preprocessor.adc import Rescaler, RescalerTargets, RescalerSettings


def plot_hist_flat(
    data_orig: Data,
    data_filt: Data,
    bins: int = 201,
    do_log: bool = False,
    do_cumulative: bool = False,
    path2save: Path | None = None,
) -> None:
    def _stats_text(x: np.ndarray) -> str:
        return (
            f"mean = {np.mean(x):.3f}\n"
            f"std = {np.std(x):.3f}\n"
            f"med = {np.median(x):.3f}\n"
            f"min = {np.min(x):.3f}\n"
            f"max = {np.max(x):.3f}\n"
            f"#smp = {x.size}"
        )

    flat_orig = np.asarray(data_orig.data).ravel()
    flat_filt = np.asarray(data_filt.data).ravel()

    fig, axs = plt.subplots(1, 2, sharey=False, figsize=(10, 4))

    for ax, x, color, title in (
        (axs[0], flat_orig, "k", "Original"),
        (axs[1], flat_filt, "r", "Rescaled"),
    ):
        ax.hist(x, bins=bins, color=color, cumulative=do_cumulative, density=do_cumulative, log=do_log, align="mid")
        ax.set_title(title)
        ax.set_xlabel(f"Signal value")
        ax.set_ylabel("Bins")
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
        method=RescalerTargets.IntSym,
        scale_safe=1.,
        total_bits=8,
        frac_bits=6,
        is_signed=False
    )

    dut = Rescaler(settings=sets)
    tdata = np.asarray(data.data, dtype=int)

    cdata = list()
    for hdata in tdata:
        out = dut(DataTransient(
            data=hdata,
            sampling_rate=data.sampling_rate
        ))
        cdata.append(out.data)
        print(dut._get_min_max(hdata))
        print(dut._scaling_factor, dut._zero_point)

    return Data(
        sampling_rate=out.sampling_rate,
        time=np.arange(start=0, stop=cdata[-1].shape[-1], step=1) / out.sampling_rate,
        data=np.asarray(cdata),
        channel_id=data.channel_id
    )


if __name__ == "__main__":
    data_orig = load_smatable_data(
        path2data=get_path_to_project() / "smatable-data" / "raw-data" / "sub-P001" / "ses-S001/knock/sub-P001_ses-S001_task-Default_run-001_knock.xdf",
        offset = 0.05
    )
    adc_lsb = 2** 17 / 1.65 * 40
    data_adc = deepcopy(data_orig)
    data_adc.data = data_orig.data * adc_lsb

    data_proc = process_data(data_adc)
    plot_hist_flat(data_orig=data_orig, data_filt=data_proc, do_log=True, do_cumulative=False)
    plot_data(data_orig=data_orig, data_filt=data_proc, yscale=1, ylabel_out="Rescaled [x LSB]")

