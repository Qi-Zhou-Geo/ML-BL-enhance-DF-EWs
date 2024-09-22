
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler, MinMaxScaler


sys.path.append(os.path.abspath("/Users/qizhou/#python/#GitHub_saved/ML-BL-enhance-DF-EWs/functions/"))
parent_dir = os.path.dirname(os.path.abspath(__file__))  # get the parent path
from dataset2dataloader import *


plt.rcParams.update( {'font.size':7, 'font.family': "Arial"} )#, 'font.weight':'bold'
input_station, feature_type, input_component = "ILL12", "C", "EHZ"

# load data
input_features_name, X_train, y_train, _, time_stamps_train = \
    select_features(input_station, feature_type, input_component, "training")

X_test = X_train.iloc[:10, :]
X_train, X_test = input_data_normalize(X_train, X_test) # the X_test is not used

df = pd.DataFrame(X_train)
df_label = pd.DataFrame(y_train)

#for i in range(len(df.columns.values)):
    #print(i, df.columns.values[i])


# <editor-fold desc="cal all corr">
def spearmanCorr(array1, array2):
    spearman_corr, p_value = spearmanr(array1, array2)
    #spearman_corr, p_value = pearsonr(array1, array2)
    return spearman_corr, p_value

featureSize = len(df.columns.values)
# to store the spearman_corr
squared_matrix_corr = np.full((featureSize, featureSize), np.nan)
squared_matrix_pValue = np.full((featureSize, featureSize), np.nan)

squared_matrix_corr1 = np.full((featureSize, featureSize), np.nan)
squared_matrix_pValue1 = np.full((featureSize, featureSize), np.nan)


for x in range(featureSize):
    for y in range(featureSize):
        array1 = df.iloc[:, x]
        array2 = df.iloc[:, y]

        corr, pValue = spearmanCorr(array1, array2)
        squared_matrix_corr[featureSize - y - 1, x] = corr
        squared_matrix_pValue[featureSize - y - 1, x] = pValue

        squared_matrix_corr1[x, y] = corr
        squared_matrix_pValue1[x, y] = pValue


# only take the half part of the matrix
#squared_matrix_corr2 = np.tril(squared_matrix_corr1, k=0)
#squared_matrix_corr2[squared_matrix_corr2==0] = np.nan

# insert column and row to seperate each features family
squared_matrix_corr_insert = squared_matrix_corr1
for step in [11, 36+1, 53+2, 70+3]:
    squared_matrix_corr_insert = np.insert(squared_matrix_corr_insert, step, np.nan, axis=1)
    squared_matrix_corr_insert = np.insert(squared_matrix_corr_insert, step, np.nan, axis=0)


def feature_corr(id1, id2):
    '''
    :param id1: feature ID
    :param id2: feature ID
    :return:
    '''
    label = np.array(df_label.iloc[:, 0], dtype=float)
    label0ID = np.where(label == 0)[0]
    label1ID = np.where(label == 1)[0]

    scaler = MinMaxScaler()
    data = np.array(df.iloc[:, id1]).reshape(-1, 1)
    data[data<0] = np.mean(data)
    x = scaler.fit_transform(data).reshape(-1)

    scaler = MinMaxScaler()
    data = np.array(df.iloc[:, id2]).reshape(-1, 1)
    data[data < 0] = np.mean(data)
    y = scaler.fit_transform(data).reshape(-1)

    spearman_corr, p_value = spearmanCorr(x, y)

    plt.scatter(x[label1ID], y[label1ID], s=3, color="#EF8636", alpha=0.1, zorder=2, label="DF")
    plt.scatter(x[label0ID], y[label0ID], s=3, color="grey", alpha=0.1, zorder=1, label="Non-DF")


    return spearman_corr
# </editor-fold>



colors = [(0, 1, 0), (1, 1, 1), (1, 0, 0)]  # RGB values (0 to 1)p
cmap = LinearSegmentedColormap.from_list('green_white_red', colors, N=256)

xLocation = np.array([0+0.5, 10+0.5, 20+1+0.5, 30+1+0.3, 40+2+0.5, 50+2+0.5, 60+2+0.5, 70+4+0.5])
xTicks = [ str(i) for i in [0, 10, 20, 30, 40, 50, 60, 70] ]

yLocation = np.array([0+0.5, 10+0.5, 20+1+0.5, 30+1+0.3, 40+2+0.5, 50+2+0.5, 60+2+0.5, 70+4+0.5])
yTicks = [ str(i) for i in [0, 10, 20, 30, 40, 50, 60, 70] ]



fig = plt.figure(figsize=(5.2, 4))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

ax0 = plt.subplot(gs[:, :2])

# <editor-fold desc="configure color">
colors = [(0, 1, 0), (1, 1, 1), (1, 0, 0)]  # RGB values (0 to 1)p
cmap = LinearSegmentedColormap.from_list('green_white_red', colors, N=256)

xLocation = np.array([0+0.5, 10+0.5, 20+1+0.5, 30+1+0.3, 40+2+0.5, 50+2+0.5, 60+2+0.5, 70+4+0.5])
xTicks = [ str(i) for i in [0, 10, 20, 30, 40, 50, 60, 70] ]

yLocation = np.array([0+0.5, 10+0.5, 20+1+0.5, 30+1+0.3, 40+2+0.5, 50+2+0.5, 60+2+0.5, 70+4+0.5])
yTicks = [ str(i) for i in [0, 10, 20, 30, 40, 50, 60, 70] ]
# </editor-fold>


# <editor-fold desc="corr heatmap">
heatmap = sns.heatmap(squared_matrix_corr_insert, annot=False, vmin=-1, vmax=1,
                      cmap="RdBu_r", square=True, cbar=False)#"RdBu_r"'inferno'

for step in [11+0.5, 36+1+0.5, 53+2+0.5, 70+3+0.5]:
    plt.axvline(x=step, color="grey", lw=1, ls="--")
    plt.axhline(y=step, color="grey", lw=1, ls="--")

plt.xlabel(f"Feature Index (ID)", weight='bold')
plt.ylabel(f"Feature Index (ID)", weight='bold')
plt.xticks(xLocation, xTicks, rotation=0, ha="center")
plt.yticks(yLocation, yTicks)

plt.title(f'(a)', weight="bold", loc='left')


ax0.xaxis.set_ticks_position('bottom')
ax0.xaxis.set_label_position('bottom')

divider = make_axes_locatable(ax0)
cax = divider.append_axes("bottom", size="2%", pad=0.45)
cbar = plt.colorbar(mappable=heatmap.get_children()[0], cax=cax, orientation="horizontal")
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.set_label('Spearman Correlation Coefficient ' + r"$\rho$", labelpad=6)
# </editor-fold>


# <editor-fold desc="positive">
ax1 = plt.subplot(gs[2])
id1, id2 = 35, 23
spearman_corr = feature_corr(id1, id2)
plt.ylim(0.5, 1.05)
plt.xlim(1e-5, 2.5)
plt.xscale("log")
#plt.ylim(-0.05, 1.05)
#plt.xlabel(f"IQR (ID {id1})", fontweight='bold')
#plt.ylabel(r"$\mathbf{ES}_{15-25}$" + f" (ID {id2})", fontweight='bold')
plt.xlabel(f"ID {id1}", fontweight='bold')
plt.ylabel(f"ID {id2}", fontweight='bold')
plt.text(x=ax1.get_xlim()[0], y=1, s=" (b)",  fontsize=6, fontweight='bold')
plt.text(x=ax1.get_xlim()[0], y=0.95, s=r" $\rho=$"+ f"{squared_matrix_corr1[id1, id2]:.3f}",  fontsize=6)
#plt.yticks([0, 0.25, 0.5, 0.75, 1.0], ["0.0", "0.25", "0.50", "0.75", "1.0"])
#plt.xticks([1e-4, 1e-2, 1e0], [r"$10^{-4}$", r"$10^{-2}$", r"$10^{0}$"])

legend = plt.legend(loc="lower right", fontsize=6)
for text in legend.get_texts():
    if text.get_text() == "DF":
        text.set_color("#EF8636")
    elif text.get_text() == "Non-DF":
        text.set_color("grey")
# </editor-fold>


# <editor-fold desc="negative">
ax2 = plt.subplot(gs[5])
id1, id2 = 47, 56
spearman_corr = feature_corr(id1, id2)
plt.ylim(1e-5, 2)
plt.xlim(-0.05, 1.05)
plt.yscale("log")
#plt.xlabel(f"Energy in [1/4, 1/2] Nyf. (ID {id1})", fontweight='bold')
#plt.ylabel(f"Mean ratio between\nMax and Median\nof all DFTs (ID {id2})", fontweight='bold')
plt.xlabel(f"ID {id1}", fontweight='bold')
plt.ylabel(f"ID {id2}", fontweight='bold')
plt.text(x=ax2.get_xlim()[0], y=3e-4, s=" (c)",  fontsize=6, fontweight='bold')
plt.text(x=ax2.get_xlim()[0], y=1e-4, s=r" $\rho=$"+ f"{squared_matrix_corr1[id1, id2]:.3f}",  fontsize=6)
#plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ["0.0", "0.25", "0.50", "0.75", "1.0"])
#plt.yticks([0, 0.25, 0.5, 0.75, 1.0], ["0.0", "0.25", "0.50", "0.75", "1.0"])
# </editor-fold>


plt.tight_layout()
plt.savefig(f"{parent_dir}/spearman_heatmap.png", dpi=600)
plt.show()

