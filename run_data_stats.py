from compute_indicators_labels_lib import get_dataset
from config import RUN as run_conf
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14

df = get_dataset(run_conf)
df["Date"] = pd.to_datetime(df["Date"])
df = df["Date"]

df = pd.DataFrame(df, columns=['Date'])

# df = df.resample('Q', on='Date').count()

Y = list(df["Date"])
x_lab = list(df.index)

fig, ax = plt.subplots(1, 1)
box = ax.get_position()
box.y0 = box.y0 + 0.1
ax.set_position(box)

ax.hist(Y, bins=66)

fig.set_size_inches(8, 6)
# plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
a = ax.get_xticklabels()

    
ax.set_facecolor('#eeeeee')
ax.set_xlabel("Year")
ax.set_ylabel("N. Samples")
ax.set_title("Data points distribution over time")

fig.savefig(run_conf["reports"] + "data_histogram.png")
plt.show()


