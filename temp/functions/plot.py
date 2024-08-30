import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

# Ruler BL Plot
def make_subplot(st:Stream, ruler:int,outdir:str, DataStart:UTCDateTime, DF_df:pd.DataFrame | pd.Series, 
                DataDuration:float, x_interval:float, BL_data: pd.DataFrame, 
                KS_thr:float= 0.95, MW_thr:float= 0.95) -> None:
    fig, ax = plt.subplots(nrows=4, ncols=1, sharex= True)
    fig.set_figheight(14)
    fig.set_figwidth(7)
    x_intervals = np.linspace(0, len(st[0].data), len(BL_data))

    # Amplitude Plot
    ax[0].plot(st[0].data, color="black")
    ax[0].set_xlim(0, len(st[0].data))
    ax[0].ticklabel_format(axis='y', style='scientific', scilimits=(-1, 3))
    ax[0].axes.xaxis.set_ticklabels([])
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(3600 * st[0].stats.sampling_rate * 1))  # unit is saecond
    ax[0].set_ylabel(f"Amplitude (nm/s)", fontweight='bold');
    xLocation = np.arange(0, (DataDuration + x_interval), x_interval)  # in hours
    xLocation1 = np.arange(0, len(st[0].data), (len(st[0].data)-1)/(DataDuration/x_interval))
    xTicks = [(UTCDateTime(DataStart) + i * 3600).strftime('%Y-%m-%d' + '\n' + '%H:%M:%S') for i in xLocation]
    if len(xLocation1) == len(xTicks):
        ax[0].set_xticks(xLocation1, xTicks)
    else:
        xTicks = xTicks[:-1]
        ax[0].set_xticks(xLocation1, xTicks)
    ax[0].hlines([-ruler,ruler], xmin = 0, xmax= (st[0].stats.endtime - st[0].stats.starttime)*st[0].stats.sampling_rate, linestyles="--",colors="r")

    ax[0].set_ylabel("Amplitude (nm/s)", fontweight='bold')
    # ax[0].set_ylim(-5e4, 5e4)

    #Goodness Plot
    ax[3].set_title("Goodness")
    ax[3].scatter(x_intervals, BL_data['goodness'].astype(float), marker=".")
    ax[3].set_ylabel("Goodness of fit (%)", fontweight='bold')
    # ax[3].set_ylim(-100,100)
    

    # Alpha Plot
    ax[2].set_title("Alpha")
    ax[2].scatter(x_intervals, BL_data['alpha'].astype(float), marker=".")
    ax[2].set_ylabel(r"Power law exponent $\alpha$", fontweight='bold')
    # ax[2].set_ylim(0, 10)


    # Follow Plot
    ax[1].set_title("Follow")
    ax[1].scatter(x_intervals, BL_data['followOrNot'].astype(float), label="Hypothesis")
    ax[1].plot(x_intervals, BL_data['ks'].astype(float), "r-", label="KS", alpha=0.6)
    ax[1].hlines(KS_thr, xmin = 0, xmax= (st[0].stats.endtime - st[0].stats.starttime)*st[0].stats.sampling_rate, linestyles="-.",colors="r")
    ax[1].hlines(MW_thr, xmin = 0, xmax= (st[0].stats.endtime - st[0].stats.starttime)*st[0].stats.sampling_rate, linestyles="--",colors="g")
    ax[1].plot(x_intervals, BL_data['MannWhitneU'].astype(float), "g-", label="MannWhitneU", alpha=0.6)
    ax[1].set_ylabel("Hypothesis Test", fontweight='bold')
    ax[1].set_ylim(-0.1,1.1)

    if type(DF_df) == pd.Series:
        DF_start = UTCDateTime(DF_df['Start(UTC+0)'])
        DF_end = UTCDateTime(DF_df[' End(UTC+0)'])
        ax[0].axvline(x=(DF_start - DataStart)* st[0].stats.sampling_rate,
                color="black", lw=2, ls="--", label="Event start time")
        ax[0].axvline(x=(DF_end - DataStart)* st[0].stats.sampling_rate,
                color="grey", lw=2, ls="--", label="Event end time")
        ax[1].axvline(x=(DF_start - DataStart)* st[0].stats.sampling_rate,
                color="black", lw=2, ls="--", label="Event start time")
        ax[1].axvline(x=(DF_end - DataStart)* st[0].stats.sampling_rate,
                color="grey", lw=2, ls="--", label="Event end time")
        ax[2].axvline(x=(DF_start - DataStart)* st[0].stats.sampling_rate,
                color="black", lw=2, ls="--", label="Event start time")
        ax[2].axvline(x=(DF_end - DataStart)* st[0].stats.sampling_rate,
                color="grey", lw=2, ls="--", label="Event end time")
        ax[3].axvline(x=(DF_start - DataStart)* st[0].stats.sampling_rate,
                color="black", lw=2, ls="--", label="Event start time")
        ax[3].axvline(x=(DF_end - DataStart)* st[0].stats.sampling_rate,
                color="grey", lw=2, ls="--", label="Event end time")
    if type(DF_df) == pd.DataFrame:
        for idx, row in DF_df.iterrows():
            DF_start = UTCDateTime(row['Start(UTC+0)'])
            DF_end = UTCDateTime(row[' End(UTC+0)'])
            ax[0].axvline(x=(DF_start - DataStart)* st[0].stats.sampling_rate,
                    color="black", lw=2, ls="--", label="Event start time")
            ax[0].axvline(x=(DF_end - DataStart)* st[0].stats.sampling_rate,
                    color="grey", lw=2, ls="--", label="Event end time")
            ax[1].axvline(x=(DF_start - DataStart)* st[0].stats.sampling_rate,
                    color="black", lw=2, ls="--", label="Event start time")
            ax[1].axvline(x=(DF_end - DataStart)* st[0].stats.sampling_rate,
                    color="grey", lw=2, ls="--", label="Event end time")
            ax[2].axvline(x=(DF_start - DataStart)* st[0].stats.sampling_rate,
                    color="black", lw=2, ls="--", label="Event start time")
            ax[2].axvline(x=(DF_end - DataStart)* st[0].stats.sampling_rate,
                    color="grey", lw=2, ls="--", label="Event end time")
            ax[3].axvline(x=(DF_start - DataStart)* st[0].stats.sampling_rate,
                    color="black", lw=2, ls="--", label="Event start time")
            ax[3].axvline(x=(DF_end - DataStart)* st[0].stats.sampling_rate,
                    color="grey", lw=2, ls="--", label="Event end time")
            
    ax[0].legend(loc='best', fontsize=6)
    ax[1].legend(loc='best', fontsize=6)

    fig.tight_layout()
    fig.savefig(f"{outdir}/image.png", dpi=300)
    plt.close()
