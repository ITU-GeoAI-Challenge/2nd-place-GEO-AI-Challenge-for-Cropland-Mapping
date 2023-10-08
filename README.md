# GEO-AI Challenge for Crop Mapping by ITU - Antoine Saget's submission

## Installation

There are two ways to run the notebooks:
- Using Google Colab
- Using a local python environment

### Using Google Colab

- []

## Files description

We provide two jupyter notebooks:
- `full_study.ipynb`: the full study. This notebook use the extra python files in `src/` and the code is not very neat as it was used in the exploratory pahse. This study consist of 6 parts:
    1. Downloading the data from GEE
    2. Data preprocessing
    3. Study on the impact of timerange
    4. Study on the impact of Sentinel-2 radiometric bands
    5. Reproduction of the submitted solution
    6. Discussion on strengths, weaknesses and possible improvements
- `simple_reproduction.ipynb`: a simplified version only reproducing the submitted solution. This notebook is self-contained and does not require any extra file. The code is more streamlined and probably easier to integrate in a production pipeline.


- `data/`: contains pre-downloaded GEE timeserie data for convenience. The data can be downloaded again by removing the files in this folder and running the notebooks. Redownloading the data from scratch will take a up to 1h30min.
- `submission/`: contains the `original_challenge_submission.csv` that has been submitted to the challenge. After running the notebooks and reproducing the solution, other `.csv` submisssion files will be created in this folder.
- `requirements.txt`: contains the list of python packages required to run the notebooks.
- `src/`: contains extra python files used in the `full_study.ipynb` notebook. None of them are required if you only want to reproduce the submitted solution. The files are:
    - `downloader.py`: Downloader class to download data from GEE
    - `dataset.py`: 
        - Dataset and Dataset_country  class that handle data loading and easy access to per country subsets
        - Dataset_training_ready and Dataset_training_ready_country class that handle data preprocessing and easy access to per country subsets
    - `model.py`: Model class that handle model training and prediction in various configurations such as per country models or global model
    - `utils.py`: various utility functions 
    - `constants.py`