from multiprocessing import freeze_support
from compute_indicators_labels_lib import preprocess
from config import RUN as run_conf

# preprocess coins timeseries and buildup the dataset with features and different
# labelings for differenti forward and backward windows combination

if __name__ == "__main__":   
    freeze_support()
    preprocess(run_conf)
