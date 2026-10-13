# Dataset

The original financial dataset is not included in this repository.

To run the experiments, place the local dataset at:

`data/raw/data_t.csv`

The notebooks expect a CSV file with an `openTime` column containing timestamps, asset-level return columns and additional market features. The PCA experiment also requires close-price columns.

The dataset is excluded from version control via `.gitignore`.

Before running experiments, verify that timestamps are chronologically ordered and that all required columns are available.