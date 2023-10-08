import pandas as pd

from src.constants import (
    ID, LON, LAT, TIMESTAMP, TARGET, IS_TRAIN, 
    COUNTRY, COUNTRY_NAME, BOUNDS, START_DATE, END_DATE,
    COLLECTION_NAME, ALL_BANDS, SCL, NDVI, B2, B3, B4, B8, 
)

from src.utils import (
    get_bounds, get_center, str_coord, filter_by_bounds, filter_by_country,
    filter_by_dates, interpolate_ts
)

from src.downloader import Downloader

class Dataset:
    """
    This class is a wrapper around a pandas Dataframe with a simple interface to access
    train, test and train_test dataframes as well as by country subdatasets.

    Internally, the class manage :
    - self._df : the train and test data with targets, lon and lat loaded from csv files
    - self._optical_df : the optical data loaded from GEE
    - self._countries : a list of subsets, each representing the data of a single country
    """
    def __init__(self, df, name, country_settings, optical_data=None, debug_level=2):
        self._df = df
        self.name = name

        self._country_settings = country_settings
        
        self._debug_level = debug_level

        # We create a subdataset for each country filtered by bounds
        self._countries = []
        for country in self._country_settings.values():
            mask = filter_by_bounds(self._df, country[BOUNDS])
            # A convenient column to distinguish between countries
            # Please see https://github.com/pandas-dev/pandas/issues/55025 if you have a warning from pandas here.
            self._df.loc[mask, COUNTRY] = country[COUNTRY_NAME]

            self._countries.append(Dataset_country(self, country))

        self.display_info()

    def from_files(train_file, test_file, name, country_settings, debug_level=2):
        """
        Create a Dataset object from challenge train and test csv files.
        """
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)

        # A single pandas Dataframe for test and train with Target at null for test
        df = pd.concat([train, test])

        # A convenient column to distinguish between train and test
        df[IS_TRAIN] = df[TARGET].notnull()

        debug_lvl = debug_level
        ds = Dataset(df, name, country_settings, debug_level=debug_lvl)
        
        return ds

    def load_all_optical_data(self, project_name, debug=False):
        """
        Load all optical data from GEE for the Dataset.
        If data is already available locally, it will be loaded from there.
        Please see Download class for more information.
        """
        downloader = Downloader(project_name, COLLECTION_NAME)
        self._optical_df = downloader.load(self, 
            ALL_BANDS, passed = pd.Timedelta(days=365), future = pd.Timedelta(days=365)
        )

        if debug:
            self._check_match_between_optical_and_train_test()
            self.display_info()

    @property
    def train(self, country=None):
        """
        Dynamically filter the dataframe to return only the train subset.
        """
        return self._df.loc[self._df[IS_TRAIN]]

    @property
    def test(self, country=None):
        """
        Dynamically filter the dataframe to return only the test subset.
        """
        return self._df.loc[~self._df[IS_TRAIN]]

    @property
    def train_test(self, country=None):
        """
        Return the full dataframe.
        """
        return self._df

    @property
    def center(self, country=None):
        """
        Return the geographical center of the dataframe.
        """
        return get_center(self.train_test)

    @property
    def bounds(self, country=None):
        """
        Return the geographical bounds of the dataframe.
        """
        return get_bounds(self.train_test)

    @property
    def countries(self):
        return self._countries


    def copy(self, debug_level=0,):
        return Dataset(
            self._df.copy(),
            self.name, 
            self._country_settings,
            self._optical_df.copy(),
            debug_level,
        )

    def display_info(self):
        """
        Display information about the dataset and by country subdatasets.
        """

        if self._debug_level > 0:
            print('#'*40)
            print('Dataset info:')
            print('    Name :', self.name)
            print('    Train shape:', self.train.shape)
            print('    Test shape:', self.test.shape)
            print('    TrainTest shape:', self.train_test.shape)

            print(f'    Center : {str_coord(self.center)}')
            bounds = self.bounds
            print('    Bounds :')
            print(f'        min : {str_coord(bounds[0])}')
            print(f'        max : {str_coord(bounds[1])}')

            # Head and info of the train and test dataframe subsets
            for subset, subset_name in zip([self.train, self.test], ['Train', 'Test']):
                print(f'    {subset_name} head:')
                display(subset.head())
                
                if self._debug_level > 1:
                    display(subset.info())

            # Head and info of the optical dataframe if available
            if hasattr(self, '_optical_df'):
                print('#'*40)
                print('Optical dataset info:')
                print('    Shape:', self._optical_df.shape)
                print('    Head:')
                display(self._optical_df.head())

                if self._debug_level > 1:
                    display(self._optical_df.info())

            print('#'*40)
            # Per country info
            for country in self.countries:
                print(f'Country {country.name} info:')
                print('    Start date:', country.start_date)
                print('    End date:', country.end_date)
                print('    Train shape:', country.train.shape)
                print('    Test shape:', country.test.shape)
                if hasattr(self, '_optical_df'):
                    print('    Optical Start date:', country.optical_start_date) 
                    print('    Optical End date:', country.optical_end_date)
                    print('    N days:', (pd.to_datetime(country.optical_end_date) - pd.to_datetime(country.optical_start_date)).days)
                    # print('    N timesteps average:', country.n_timesteps_average())   
                    print(f'    Mem usage: {country.optical_df.memory_usage().sum() / 1024**2:.0f} MB')

        if self._debug_level > 2:
            self._check_for_duplicates()

    def _check_for_duplicates(self):
        """
        This method check for rows in the dataframe with the same (lon, lat) pair.
        """
        print(f'{self._df[[LON, LAT]].duplicated(keep=False).sum()} duplicates found in dataset :')
        display(self._df.loc[self._df[[LON, LAT]].duplicated(keep=False), [ID, LON, LAT, TARGET]].sort_values(by=[LON, LAT]))

    def _check_match_between_optical_and_train_test(self):
        """
        This method check that the optical dataframe matches the normal df in terms of IDs, lon and lat.
        """
        for idd, lon, lat in zip(self.train_test[ID], self.train_test[LON], self.train_test[LAT]):
            tmp = self._optical_df.loc[idd == self._optical_df[ID]]
            
            lons, lats = tmp[LON].unique(), tmp[LAT].unique()
            if len(lons) != 1 or len(lats) != 1:
                print(f'Error: {len(lons)} lons and {len(lats)} lats found for {idd}')

            if lon != lons[0] or lat != lats[0]:
                print(f'Error: {lon} lon and {lat} lat found for {idd} in train_test but {lons[0]} lon and {lats[0]} lat found in optical_df')

class Dataset_country(Dataset):
    """
    Dataset_country class is a subdataset of a Dataset object filtered by country bounds
    it allows to access the train, test and train_test dataframes with 
    the same interface as the Dataset class by dynamically filtering the
    dataframes with the country bounds on each access.
    """
    def __init__(self, ds, country, debug_level=0):
        self._ds = ds
        self._filter_bounds = country[BOUNDS]
        self.start_date = country[START_DATE]
        self.end_date = country[END_DATE]
        self.name = country[COUNTRY_NAME]
        self._debug_level = debug_level
        self._df = ds._df.copy()

        self.display_info()

    @property
    def train(self):
        """
        Dynamically filter the main dataframe to return only the train subset of the country.
        """
        mask = filter_by_country(self._df, self)
        return self._df.loc[self._df[IS_TRAIN] & mask]

    @property
    def test(self):
        """
        Dynamically filter the main dataframe to return only the test subset of the country.
        """
        mask = filter_by_country(self._df, self)
        return self._df.loc[~self._df[IS_TRAIN] & mask]

    @property
    def train_test(self):
        """
        Dynamically filter the main dataframe to return only the subset of the country.
        """
        mask = filter_by_country(self._df, self)
        return self._df.loc[mask]

    @property
    def center(self):
        """
        Return the geographical center of the country.
        """
        mask = filter_by_country(self.train_test, self)
        return get_center(self.train_test[mask])

    @property
    def bounds(self):
        """
        Return the geographical bounds of the country.
        """
        mask = filter_by_country(self.train_test, self)
        return get_bounds(self.train_test[mask])

    @property
    def ids(self):
        """
        Return the IDs of the datapoints in the country.
        """
        return self.train_test[ID]

    @property
    def optical_df(self):
        """ 
        Return the optical dataframe filtered to only contain the data of the country.
        """
        mask = self._ds._optical_df[ID].isin(self.ids)
        return self._ds._optical_df.loc[mask]

    @property
    def optical_start_date(self):
        return self.optical_df[TIMESTAMP].min()

    @property
    def optical_end_date(self):
        return self.optical_df[TIMESTAMP].max()

class Dataset_training_ready():
    """
    This class is a wrapper around Dataset that expose a simple interface to access X_train, Y_train, and X_test.
    Most importantly, it handles the interpolation, redindexing and resampling of the optical data into
    fixed length timeseries ready for use in a scikit-learn model.
    """
    TABULAR = 'tabular'
    TIMESERIES = 'timeseries'

    def __init__(self, df, df_optical, bands, datatype, debug_level=2):
        self._df = df
        self._df_optical = df_optical

        self.bands = bands
        self._data_type = datatype

        self.countries = []
        for country_name in self._df[COUNTRY].unique():
            self.countries.append(Dataset_training_ready_country(self, country_name))

        if debug_level > 0:
            display(self._df.head())
        if debug_level > 1:
            display(self._df.info())

        if debug_level > 0:
            display(self._df_optical.head())
        if debug_level > 1:
            display(self._df_optical.info())

    def get_ts_data_from(ds, bands, project_name, only_country=None, start_date=None, end_date=None, country_settings=None):
        """
        Given a Dataset and other parameters, this function will create a Dataset_training_ready object 
        with X_train, Y_train, X_test ready for use in a scikit-learn model.
        """
        ds.load_all_optical_data(project_name)

        if only_country is not None:
            df = ds._df.loc[ds._df[COUNTRY] == only_country].copy().set_index(ID)
        else:
            df = ds._df.copy().set_index(ID)

        df_optical = pd.DataFrame()

        if only_country is not None:
            for country in ds.countries:
                if country.name == only_country:
                    optical = filter_by_dates(country.optical_df, start_date, end_date)
                    optical = optical[[ID, TIMESTAMP] + bands]
                    df_optical = pd.concat([df_optical, optical])
        else :            
            for country in ds.countries:
                if country_settings is None:
                    start_date = country.start_date
                    end_date = country.end_date
                else:
                    start_date = country_settings[country.name][START_DATE]
                    end_date = country_settings[country.name][END_DATE]
                optical = filter_by_dates(country.optical_df, start_date, end_date)
                optical = optical[[ID, TIMESTAMP] + bands]

                df_optical = pd.concat([df_optical, optical])
            
        if SCL in bands:
            for scl in df_optical[SCL].unique():
                SCL_COL = f'{SCL}_{scl}'
                # add collumn
                df_optical[SCL_COL] = df_optical[SCL] == scl
                df_optical[SCL_COL] = df_optical[SCL_COL].astype('uint8')
                bands.append(SCL_COL)
            bands.remove(SCL)
            df_optical = df_optical[[ID, TIMESTAMP] + bands]
        

        return Dataset_training_ready(df, df_optical, bands, Dataset_training_ready.TIMESERIES, debug_level=0)

    @property
    def X_train(self):
        """
        Dynamically reindex, interpolate and resample the optical train data into as (n_samples, n_timesteps*n_bands) array.
        """
        if self._data_type == Dataset_training_ready.TABULAR:
            return self._df_optical.loc[self._df[IS_TRAIN], self.bands].sort_index()
        elif self._data_type == Dataset_training_ready.TIMESERIES:
            ids = self._df.loc[self._df[IS_TRAIN]].index
            optical_filtered = self._df_optical.loc[self._df_optical[ID].isin(ids)]
            X, IDs = interpolate_ts(optical_filtered, self.bands, start_date=self._df_optical[TIMESTAMP].min(), end_date=self._df_optical[TIMESTAMP].max())
            X = X.reshape(X.shape[0], -1)
            return pd.DataFrame(X, index=IDs).sort_index()

    @property
    def Y_train(self):
        """
        Return the target values of the train subset, matching the order of X_train.
        """
        return self._df.loc[self._df[IS_TRAIN], TARGET].sort_index().astype('uint8')

    @property
    def X_test(self):
        """
        Dynamically reindex, interpolate and resample the optical test data into as (n_samples, n_timesteps*n_bands) array.
        """
        if self._data_type == Dataset_training_ready.TABULAR:
            return self._df_optical.loc[~self._df[IS_TRAIN], self.bands].sort_index()
        elif self._data_type == Dataset_training_ready.TIMESERIES:
            ids = self._df.loc[~self._df[IS_TRAIN ]].index
            optical_filtered = self._df_optical.loc[self._df_optical[ID].isin(ids)]
            X, IDs = interpolate_ts(optical_filtered, self.bands, start_date=self._df_optical[TIMESTAMP].min(), end_date=self._df_optical[TIMESTAMP].max())
            X = X.reshape(X.shape[0], -1)
            return pd.DataFrame(X, index=IDs).sort_index()

    @property
    def ids(self):
        return pd.Series(self._df.index)

    def get_baseline_data_from(ds, project_name):
        ds.load_all_optical_data(project_name)

        bands = [B2, B3, B4, B8, NDVI]

        df = ds._df.copy().set_index(ID)

        df_optical = pd.DataFrame(columns=bands)
        for country in ds.countries:
            optical = filter_by_dates(country.optical_df, country.start_date, country.end_date)
            optical = optical[[ID] + bands]
            optical = optical.groupby(ID).mean()

            df_optical = pd.concat([df_optical, optical])

        return Dataset_training_ready(df, df_optical, bands, Dataset_training_ready.TABULAR)

class Dataset_training_ready_country(Dataset_training_ready):
    """
    Dataset has a list of Dataset_country, Dataset_training_ready has a list of Dataset_training_ready_country.
    This class is a subdataset of a Dataset_training_ready object filtered by country
    """
    def __init__(self, ds, country_name):
        self._ds = ds
        self.country_name = country_name

    def country_mask(self):
        return self._ds._df[COUNTRY] == self.country_name

    @property
    def X_train(self):
        """
        Dynamically reindex, interpolate and resample the country optical train data into as (n_samples, n_timesteps*n_bands) array.
        """
        if self._ds._data_type == Dataset_training_ready.TABULAR:
            return self._ds.X_train.loc[self.country_mask()]
        elif self._ds._data_type == Dataset_training_ready.TIMESERIES:
            ids = self._ds._df.loc[self._ds._df[IS_TRAIN] & self.country_mask()].index
            optical_filtered = self._ds._df_optical.loc[self._ds._df_optical[ID].isin(ids)]
            X, IDs = interpolate_ts(optical_filtered, self._ds.bands, start_date=self._ds._df_optical[TIMESTAMP].min(), end_date=self._ds._df_optical[TIMESTAMP].max())
            X = X.reshape(X.shape[0], -1)
            return pd.DataFrame(X, index=IDs).sort_index()

    @property
    def Y_train(self):
        """
        Return the target values of the country train subset, matching the order of X_train.
        """
        return self._ds.Y_train.loc[self.country_mask()].astype('uint8')

    @property
    def X_test(self):
        """
        Dynamically reindex, interpolate and resample the country optical test data into as (n_samples, n_timesteps*n_bands) array.
        """
        if self._ds._data_type == Dataset_training_ready.TABULAR:
            return self._ds.X_test.loc[self.country_mask()]
        elif self._ds._data_type == Dataset_training_ready.TIMESERIES:
            ids = self._ds._df.loc[~self._ds._df[IS_TRAIN ] & self.country_mask()].index
            optical_filtered = self._ds._df_optical.loc[self._ds._df_optical[ID].isin(ids)]
            X, IDs = interpolate_ts(optical_filtered, self._ds.bands, start_date=self._ds._df_optical[TIMESTAMP].min(), end_date=self._ds._df_optical[TIMESTAMP].max())
            X = X.reshape(X.shape[0], -1)
            return pd.DataFrame(X, index=IDs).sort_index()
    
    @property
    def ids(self):
        mask = self.country_mask()
        return pd.Series(self._ds._df.loc[mask].index)
