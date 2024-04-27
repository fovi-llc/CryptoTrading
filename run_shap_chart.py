import pickle
from matplotlib import pyplot as plt
import shap
from config import RUN as run_conf
import sys

handle = open('explainer_5_2_50000.pickle', 'rb')
ex = pickle.load(handle)

# f_name = ['Z_score', 'RSI', 'boll', 'ULTOSC', 'pct_change', 'zsVol', 'PR_MA_Ratio_short', 'MA_Ratio_short', 'MA_Ratio', 'PR_MA_Ratio', 'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLBELTHOLD', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGRAVESTONEDOJI', 'CDLHANGINGMAN', 'CDLHARAMICROSS', 'CDLINVERTEDHAMMER', 'CDLMARUBOZU', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLPIERCING', 'CDLRISEFALL3METHODS', 'CDLSHOOTINGSTAR', 'CDLSPINNINGTOP', 'CDLUPSIDEGAP2CROWS', 'DayOfWeek', 'Month', 'Hourly']

f_name = ['ZScore', 'RSI', 'Bollinger', 'Ultosc', 'PctChange', 'ZScoreVol', 'EmaCross1_21', 'EmaCross21_50', 'EmaCross50_100', 'EmaCross1_50', 'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLBELTHOLD', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGRAVESTONEDOJI', 'CDLHANGINGMAN', 'CDLHARAMICROSS', 'CDLINVERTEDHAMMER', 'CDLMARUBOZU', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLPIERCING', 'CDLRISEFALL3METHODS', 'CDLSHOOTINGSTAR', 'CDLSPINNINGTOP', 'CDLUPSIDEGAP2CROWS', 'DayOfWeek', 'Month', 'Hourly']

ex.feature_names = f_name


max_feat = 11

fig = plt.figure()
shap.plots.beeswarm(ex, max_display=max_feat, show=False)
plt.gcf().set_size_inches(8, 8)
plt.title(label="")
ax = fig.axes[0]
box = ax.get_position()
box.x0 = box.x0 + 0.2
ax.set_position(box)

fig.savefig(run_conf["reports"] + "shap_beeswarm.png")

plt.show()

fig = plt.figure()
shap.plots.bar(ex, max_display=max_feat, show=False)
plt.gcf().set_size_inches(8, 8)
plt.title(label="")

ax = fig.axes[0]
box = ax.get_position()
box.x0 = box.x0 + 0.15
box.y1 = box.y1 - 0.05
ax.set_position(box)
fig.savefig(run_conf["reports"] + "shap_importance.png")
plt.show()


# waterfall
for i in range(0, 10):
    fig = plt.figure()
    shap.plots.waterfall(ex[i], max_display=max_feat)
    fig.savefig(run_conf["reports"] + "shap_waterfall_%d.png" % i)
    plt.show()


# top 5
for f in ("Bollinger", "EmaCross1_21", "RSI", "ZScore", "EmaCross1_50"):
    plt.figure()
    shap.plots.scatter(ex[0:10000, f], color=ex[0:10000], alpha=0.25, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    ax = fig.axes[0]
    box = ax.get_position()
    box.x0 = box.x0 + 0.02
    ax.set_position(box)
    fig.savefig(run_conf["reports"] + "shap_dep_%s.png" % f)
    plt.show()
