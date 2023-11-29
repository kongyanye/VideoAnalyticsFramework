import matplotlib.pyplot as plt
import pandas as pd

# used to draw arrow
import sys

sys.path.append('../')
from utils import add_better

# style from https://github.com/garrettj403/SciencePlots
# pip install SciencePlots
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    "font.family": "sans-serif",  # specify font family here
    "font.serif": ["arial"],  # specify font here
    "font.size": 16,
    'ps.fonttype': 42
})  # specify font size here

# plot params
fontsize = 16
plt.rcParams['font.family'] = 'Arial'
colors = [
    '#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e'
]  # standard corlors from SciencePlots
markersize = 8
linestyles = ['-', '--', ':', '-.', '-']
markers = ['^', '+', 'p', 's', '<', '>', '*', 'x']
rotation = 30

# summarize results
acc = {}
size = {}
qfactors = [10, 30, 50, 70, 90]
for video in ['road.mp4', 'uav']:
    if video not in acc:
        acc[video] = []
    if video not in size:
        size[video] = []
    for qfactor in qfactors:
        filepath = f'/home/sig/files/VideoAnalyticsFramework/results/impact_of_qfactor/{video}/{qfactor}.csv'
        df = pd.read_csv(filepath, index_col=0)
        mean_size = df['size'].mean()
        mean_acc = df['map50'].mean()
        mean_lat = df['lat_infer'].mean()
        acc[video].append(mean_acc)
        size[video].append(mean_size)

# plot, use ratio of 5:3 for one figure and 10:3 for two figures
fig, axes = plt.subplots(1, 2, figsize=(10, 3), dpi=200)

# size
ax = axes[0]
for i, d in enumerate(['road.mp4', 'uav']):
    ax.plot(qfactors,
            size[d],
            marker=markers[i],
            linestyle=linestyles[i],
            lw=2,
            markersize=markersize,
            alpha=1,
            color=colors[i])

ax.legend(['Road', 'UAV'],
          prop={'size': fontsize},
          ncol=4,
          bbox_to_anchor=[0.5, 1],
          loc='lower center')

ax.set_axisbelow(True)
ax.xaxis.grid(which='major', linestyle='--')
ax.yaxis.grid(which='major', linestyle='--')
# ax.set_ylim(bottom=-0.1, top=0.8)
# ax.set_yticks(np.arange(0.2, 1.0, 0.2))
# yticklabels = np.arange(0, 1.2, 0.2)
# ax.set_yticklabels([f'{each:.2f}' for each in yticklabels], size=fontsize)
ax.set_ylabel('Size (KB)', size=fontsize)
ax.set_xlabel('QFactor', size=fontsize)
ax.set_xticks(qfactors)
# ax.set_xticklabels(labels=label, size=fontsize, rotation=rotation, ha='center')

add_better(ax, 0.5, 0.6, 'down')

# acc
ax = axes[1]
for i, d in enumerate(['road.mp4', 'uav']):
    ax.plot(qfactors,
            acc[d],
            marker=markers[i],
            linestyle=linestyles[i],
            lw=2,
            markersize=markersize,
            alpha=1,
            color=colors[i])

# you can also remove this legend and move the previous legend to the center of figure
ax.legend(['Road', 'UAV'],
          prop={'size': fontsize},
          ncol=4,
          bbox_to_anchor=[0.5, 1],
          loc='lower center')

ax.set_axisbelow(True)
ax.xaxis.grid(which='major', linestyle='--')
ax.yaxis.grid(which='major', linestyle='--')
# ax.set_ylim(bottom=-0.1, top=0.8)
# ax.set_yticks(np.arange(0.2, 1.0, 0.2))
# yticklabels = np.arange(0, 1.2, 0.2)
# ax.set_yticklabels([f'{each:.2f}' for each in yticklabels], size=fontsize)
ax.set_ylabel('mAP@50', size=fontsize)
ax.set_xlabel('QFactor', size=fontsize)
ax.set_xticks(qfactors)
# ax.set_xticklabels(labels=label, size=fontsize, rotation=rotation, ha='center')

## Tips: more lines shall be added to make the figure contain more information

add_better(ax, 0.5, 0.2, 'up')

# save to eps (vector format)
fig.savefig('./impact_of_qfactor.eps', format='eps', bbox_inches='tight')
