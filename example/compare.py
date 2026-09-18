from pathlib import Path
import numpy as np
from dataclasses import dataclass

from pyxdf import load_xdf
from matplotlib import pyplot as plt

from elasticai.preprocessor import get_path_to_project
from elasticai.preprocessor.downsampling import SettingsDownSampling, DownSampling
from elasticai.preprocessor.filter import SettingsFilter, Filtering


@dataclass
class Data:
    sampling_rate: float
    channel_id: list[int]
    time: np.ndarray
    data: np.ndarray


def load_smatable_data(path2data: Path) -> Data:
    # Download the data here to local folder: https://zenodo.org/records/20408458
    used_channels = [1, 4, 6, 8]
    streams, _ = load_xdf(
        filename=path2data.as_posix(),
        dejitter_timestamps=True,
        synchronize_clocks=False,
    )
    stream = next(
        stream
        for stream in streams
        if np.asarray(stream["time_series"]).ndim == 2
        and np.asarray(stream["time_series"]).shape[1] >= 9
    )
    scaling = 3.3 / 2 ** 17 / 40
    return Data(
        sampling_rate=float(stream["info"]["nominal_srate"][0]),
        channel_id=used_channels,
        time=np.asarray(stream["time_stamps"]-stream["time_stamps"][0], dtype=np.float32),
        data=scaling * np.asarray(stream["time_series"][:, used_channels], dtype=np.float32).T
    )


def process_data(data: Data) -> Data:
    filt = Filtering(
        settings=SettingsFilter(
            gain=1.0,
            fs=data.sampling_rate,
            n_order=1,
            f_filt=[100., 300.],
            type="iir",
            f_type="butter",
            b_type="bandpass"
        )
    )
    down = DownSampling(
        settings=SettingsDownSampling(
            sampling_rate=data.sampling_rate,
            dsr=8
        )
    )

    data0 = list()
    for idx, xraw in enumerate(data.data):
        x = filt.filt(xin=xraw)
        data0.append(down.do_decimation_polyphase(uin=x, take_first_order=True))
    return Data(
        sampling_rate=down.sampling_rate_out,
        time=np.arange(start=0, stop=data0[0].size, step=1) / down.sampling_rate_out,
        data=np.asarray(data0),
        channel_id=data.channel_id
    )


def ticks_with_zero(vmin, vmax, n: int=7):
    vmin = min(vmin, 0)
    vmax = max(vmax, 0)
    span = vmax - vmin
    if span == 0:
        return np.array([0])

    n_pos = int(round((n - 1) * vmax / span))
    n_neg = (n - 1) - n_pos
    neg = np.linspace(vmin, 0, n_neg + 1) if n_neg > 0 else np.array([0.0])
    pos = np.linspace(0, vmax, n_pos + 1) if n_pos > 0 else np.array([0.0])
    ticks = np.unique(np.concatenate([neg, pos]))
    ticks = 2 * np.round(ticks / 2)
    ticks = np.unique(ticks.astype(int))
    if 0 not in ticks:
        ticks = np.sort(np.append(ticks, 0))

    return ticks


def plot_data(data_orig: Data, data_filt: Data) -> None:
    num_rows = 2
    num_cols = 4
    scaley = 1e3
    _, axs = plt.subplots(num_rows,num_cols, sharex=True, sharey=True)

    for idx, (raw_orig, raw_filt) in enumerate(zip(data_orig.data, data_filt.data)):
        ysel = idx % num_cols
        axs[0, 0].set_ylabel("Original [mV]")
        axs[0, 0].set_xlim([data_orig.time[0], data_orig.time[-1]])
        axs[0, ysel].plot(data_orig.time, scaley * raw_orig, color="k", label=f"CH-{data_orig.channel_id[idx]:02d}")
        axs[0, ysel].grid(True)
        axs[0, ysel].legend(loc="upper right")

        axs[1, 0].set_ylabel("Filtered [mV]")
        axs[1, 2].set_xlabel("Time / s")
        axs[1, ysel].plot(data_filt.time, scaley * raw_filt, color="r", label=f"CH-{data_filt.channel_id[idx]:02d}")
        axs[1, ysel].grid(True)
        axs[1, ysel].legend(loc="upper right")

    xmin, xmax = axs[0, 0].get_xlim()
    ymin, ymax = axs[0, 0].get_ylim()
    xticks = ticks_with_zero(xmin, xmax, n=6)
    yticks = ticks_with_zero(ymin, ymax, n=7)
    axs[0,0].set_xticks(xticks)
    axs[0,0].set_yticks(yticks)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_orig = load_smatable_data(
        path2data=get_path_to_project() / "smatable-data" / "raw-data" / "sub-P001" / "ses-S001/knock/sub-P001_ses-S001_task-Default_run-001_knock.xdf"
    )
    data_proc = process_data(data_orig)
    plot_data(data_orig=data_orig, data_filt=data_proc)
