This is the code for the paper:

A profitable trading algorithm for cryptocurrencies using a Neural Network model,
Mimmo Parente, Luca Rizzuti, Mario Trerotola,
https://doi.org/10.1016/j.eswa.2023.121806 
(https://www.sciencedirect.com/science/article/pii/S0957417423023084).

Abstract: Algorithmic trading enables the execution of orders using a set of rules determined by a 
computer program. Orders are submitted based on an asset’s expected price in the future, an approach well 
suited for high-volatility markets, such as those trading in cryptocurrencies. The goal of this study is to 
find a reliable and profitable model to predict the future direction of a crypto asset’s price based on 
publicly available historical data. We first develop a novel labeling scheme and map this problem into a 
Machine Learning classification problem. The model is then validated on three major cryptocurrencies through 
an extensive backtest over a bull, bear and flat market. Finally, the contribution of each feature to the 
classification output is analyzed.

Keywords: Cryptocurrencies; Machine learning; Neural network; Price prediction; Algorithmic trading; Explainable AI; Backtesting; Shapley values

These source files are from the paper's reference to
https://figshare.com/articles/software/CryptoTrading_zip/22953377
Version 2 Software posted on 2023-05-20, 00:18 authored by Mimmo Parente, Luca Rizzuti
License GPL 3+

I've omitted the data files (sourced from Binance) and the binary weights/models/reports contained in 
the  original ZIP file in order to be conservative wrt rights and file size.  Please access the original as 
necessary for those.

I've added a requirements.txt which has some notes about how got this to work.  The main thing necessary 
besides pip installation is the binary for TA-Lib which is `brew install ta-lib` on MacOS.
See https://github.com/TA-Lib/ta-lib-python#dependencies for others.

=== ORIGINAL README ===
The software has been developed on Linux.
The code can run with a standard Python interpreter.
However, it is strongly encouraged the use of a working installation of Tensorflow *on CUDA GPU*.

Python Package Prerequisites
------------
Numpy: numpy.org
SkLearn: scikit-learn.org
matplotlib: matplotlib.org 
Tensorflow: tensorflow.org 
Shap: github.com/slundberg/shap 
Binance api: github.com/binance/binance-connector-python


Folders
-------
root folder: contains outputs from NN training and shap explanations.
reports/: contains all script outputs. 
rep_charts_paper/: contains figure, tables, etc. reported in the article
processed_data/: contains the preprocessed dataset with all the labeling schemes applied
raw_data_4_hour/: contains the raw datasets downloaded from binance API endpoint


Running the pipeline
--------------------
The pipeline code is constituted by a series of scripts to run in sequence.

- config.py: script configurations.

- run_download_data.py: to be updated with own Binance api key and secret key.
    	Creates the raw set of cryptos into the folder asset_data/raw_data_4_hour/

- run_preprocess_dataset.py: 
	Creates the preprocessed dataset and saves it into a csv file in the folder processed_data/

- run_data_stats.py:
	 Plots the charts of time data distribution.

- run_alpha_beta.py: 
	Computes alpha and beta, (the computed values must be copied and pasted into config.py).

- run_search_bw_fw.py: 
	The grid search for backward and forward windows. The output is saved into the file reports/final_ncr_1.xlsx

- run_train_final.py:
	The training of the five final models. The output saves reports into reports/final_model_*_*.xlsx. 
	One file for each backward/forward window combination.

- run_backtest_final.py:
	The backtest of the above five models and saves reports into reports/backtest_final.xlsx.
    
- run_shap_explainer.py:
	Creates and serializes on disk the SHAP explanation object.

- run_shap_chart.py:
	Draws all shap charts.
