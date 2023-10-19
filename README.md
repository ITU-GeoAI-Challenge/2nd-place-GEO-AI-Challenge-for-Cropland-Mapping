# GEO-AI Challenge for Crop Mapping by ITU - Antoine Saget's submission

This repository present the 2nd place solution for the Zindi GEO-AI Challenge for Cropland Mapping by ITU that achieve a 0.943 accuracy on the private leaderboard.

We provide two jupyter notebooks:
- `full_study.ipynb` : the full study. We recommend this notebook for people interested in the decisions that lead to the final submission. **We strongly recommend reading section 6 where results are summarized and strengths, weaknesses and possible improvements are discussed.** This notebook use the extra python files in `src/` and the code is not very neat as it was used in the exploratory phase. This study consist of 6 parts:
    - 1 - Downloading the data from GEE
    - 2 - Data preprocessing to obtain aligned timeseries of same lengths
    - 3 - Study of different time spans (length) and periods (start date)
    - 4 - Study of different Sentinel-2 radiometric bands
    - 5 - Reproduction of the submitted solution
    - 6 - Discussion on strengths, weaknesses and possible improvements
- `simple_reproduction.ipynb` : a simplified version only reproducing the submitted solution. This notebook is self-contained and does not require any extra file. The code is more streamlined and probably easier to integrate in a production pipeline.

## Solution

The solution is explained in details in `full_study.ipynb`.
A summary is given below :
- **Data**:
    - Sentinel-2 time series data is used
    - The time series lengths and periods are optimized per country with CV on the training data
    - The choice of which Sentinel-2 radiometric bands to keep is optimized with CV on the training data
- **Model**:
    - A simple shallow RandomForest per country (using python scikit-learn implementation).

## Installation

There are two ways to run the notebooks:
- With Google Colab
- Using a local python environment

### Using Google Colab

- [full_study.ipynb](https://colab.research.google.com/drive/1VNLHFsI4rpmS75hzLY9INsr7lHKNs-wB?usp=sharing) (<15min runtime on Google Colab)
- [simple_reproduction.ipynb](https://colab.research.google.com/drive/1aMbBlbrlwZMYgKggQ2BYovC9_6kvB-kR?usp=sharing) (<3min runtime on Google Colab)

### Using a local python environment

The python environment can be installed using the `requirements.txt` file. 
We recommend using a virtual environment such as `conda` or `virtualenv` to avoid conflicts with your system python installation.

```bash
python3 -m venv env 
source env/bin/activate
pip install -r requirements.txt
```

Then you can run the notebooks using `jupyter notebook` or `jupyter lab`.

## Folders descriptions

- `data/`: contains pre-downloaded GEE timeserie data for convenience.
- `submission/`: contains the `original_challenge_submission.csv` that has been submitted to the challenge. After running the notebooks and reproducing the results, other `.csv` submisssion files will be created in this folder.
- `src/`: contains extra python files used in the `full_study.ipynb` notebook. None of them are required if you only want to reproduce the submitted solution with the `simple_reproduction.ipynb` notebook.
    - `downloader.py`: Downloader class to download data from GEE
    - `dataset.py`: 
        - `Dataset` and `Dataset_country` class that handle data loading and easy access to per country subsets
        - `Dataset_training_ready` and `Dataset_training_ready_country` class that handle data preprocessing and easy access to per country subsets
    - `model.py`: `Model` class that handle model training and prediction in various configurations such as per country models or global model
    - `utils.py`: various utility functions 
    - `constants.py`

## Redownloading the data

We provide pre-downloaded Sentinel-2 GEE timeseries data for convenience in the `data/` folder.  
The data can be redownloaded from GEE by removing the files in the `data/` folder and running the notebooks.  
Re-downloading the data from scratch will take up to 1h30min.  
As the GEE requests are quite long, we use GEE tasks system. **This requires a GEE account and the user must set `PROJECT_NAME` to it's own GEE project name in the notebooks. Otherwise, the GEE download tasks will fail.**
