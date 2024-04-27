The software has ben developed on Linux.
The code can run with a standard Python interpreter.
However, it is strongly encouraged the use of
a working installation of Tensorflow *on CUDA GPU*.

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
