import ee
import time
import pandas as pd

from constants import LON, LAT, TIMESTAMP, ID, COUNTRY, RADIUS_CIRCLES, C_CRICLES, WEIGHT_CIRCLES, TS_ID
from utils import fc_from_points, get_center, get_bounds, filter_by_bounds, filter_by_dates, filter_by_country, base_folium, str_coord, fc_to_dict, addNDVI

class Downloader:
    """
    This class is used to download data from Google Earth Engine.
    The class will handle local caching of the data to avoid downloading the same data multiple times.
    See Downloader.load() for more details. 
    """
    def __init__(self, project_name, collection_name):
        """
        project_name: str
            The name of the GEE project to use.
        collection_name: str
            The name of the GEE collection to use.
        """
        self._tasks = []
        self.project_name = project_name
        self.collection_name = collection_name
        self.basename = f'projects/{self.project_name}/assets/'
        self.cache_folder = 'data/'

    def load(self, 
        ds, bands, 
        passed = pd.Timedelta(days=0), future = pd.Timedelta(days=0),
        cache=True
    ):
        """
        Load time series data from GEE.
        ds: Dataset
            The dataset to load the data for.
        bands: list of str
            The bands to load.
        passed: pd.Timedelta
            The number of days before the ds.start_date to load (used to load extra data).
        future: pd.Timedelta
            The number of days after the ds.end_date to load (used to load extra data).
        cache: bool
            Whether to cache the data locally.
        
        1. If the data is already available locally in the data/ folder, it will be loaded from there.
        2. If the data is already available on GEE, it will be loaded from there.
        3. If the data is not available locally or on GEE, download tasks will be launched on GEE, and the data will be loaded from there. 
        
        2. and 3. require a Google Earth Engine account with at least 30MB of available storage space.
        3. can take up to 1h30min to complete but only needs to be done once if cache=True.
        """

        # 1. Try to load the data from local files
        optical_df = self._load_locally(ds, bands, passed, future)
        if optical_df is not None:
            return optical_df

        # 2. Try to load the data from ee
        optical_df = self._load_from_ee(ds, bands, passed, future, cache=cache)
        if optical_df is not None:
            return optical_df

        # 3. Launch a task to download the data from ee
        optical_df = self._download(ds, bands, passed, future, cache=cache)
        if optical_df is not None:
            return optical_df

        print('#'*40)
        print('Something went wrong. No data was loaded.')
        print('#'*40)
    
        return None

    def _start_download_task(self, country, start_date, end_date, bands, filename, scale=10):
        """
        This functions is called if the data is not available locally or on GEE.
        It will launch a task on GEE to download the data.
        The GEE pipeline is defined here and might be interesting to people who want to extraxct time series data from GEE.
        """
        points_fc = fc_from_points(country.train_test)
        collection = (ee
            .ImageCollection(self.collection_name)
            .filterDate(start_date, end_date)
            .filterBounds(points_fc)
        )

        if 'NDVI' in bands:
            collection = collection.map(addNDVI)

        # This timeseries extraction technique is inspired by https://spatialthoughts.com/2020/04/13/extracting-time-series-ee/
        def image_reducer(image):
            def feature_reducer(feature):
                coordinates = feature.geometry().coordinates()
                feature_dict = {
                    'millis': image.date().millis(),
                    TS_ID: feature.id(),
                    LON: coordinates.get(0),
                    LAT: coordinates.get(1)
                }
                
                for band in bands:
                    band_filtered = ee.List([feature.get(band), -9999]).reduce(ee.Reducer.firstNonNull())
                    feature_dict[band] = band_filtered

                return feature.set(di)

            return image.select(bands).reduceRegions(
                collection=points_fc,
                reducer=ee.Reducer.first(),
                scale=scale,
            ).map(feature_reducer)
        
        stats_fc = (ee.FeatureCollection(collection
            .map(image_reducer)
            .flatten()
            .filter(ee.Filter.neq(bands[0], -9999))
            .distinct(['system:index', 'millis'])
        ))

        task = ee.batch.Export.table.toAsset(
            collection=stats_fc,
            description='stats_fc export',
            assetId=self.basename + filename,
        )
        task.start()

        return task

    def _prepare_df(self, ds, stats_dict, bands):
        """
        Once the data is downloaded from GEE, it is converted to a pandas dataframe.
        Minor preprocessing is done here: adding the date, removing useless columns, etc.
        """
        stats_df = pd.DataFrame(stats_dict)

        def add_date_info(df):
            df[TIMESTAMP] = pd.to_datetime(df['millis'], unit='ms')
            return df

        stats_df = add_date_info(stats_df)
        stats_df = stats_df.drop(columns=['millis', 'system:index'])
        
        stats_df['TS_ID'] = stats_df['TS_ID'].astype('uint')
        ts_ids = np.unique(stats_df['TS_ID'])
        assert len(ts_ids) == len(ds.train_test)
        for ts_id, id in zip(np.unique(stats_df['TS_ID']), ds.train_test[ID]):
            stats_df.loc[stats_df['TS_ID'] == ts_id, ID] = id
        stats_df = stats_df.drop(columns=['TS_ID'])
            
        stats_df.sort_values(by=[ID, TIMESTAMP], inplace=True)

        return stats_df

    def _download(self, ds, bands, passed, future, cache=True):
        """
        This function is called if the data is not available locally.
        The function willl try to load the data from GEE.
        If not available on GEE, it will launch tasks to download the data from GEE and wait for the tasks to complete.
        """
        tasks = []
        dfs = []
        for country in ds.countries:
            start_date, end_date = self._get_offset_dates(
                country.start_date, country.end_date,
                passed, future
            )
            filename = self._get_filename(country, start_date, end_date, bands)

            try:
                df = self._load_df_country(country, filename, bands, cache)
                dfs.append(df)
                continue

            except ee.EEException as e:
                print(e)
                print(f'File for {country.name} does not exist. Downloading...')
    
                task = self._start_download_task(country, start_date, end_date, bands, filename)
                tasks.append([task, country.name])

        if len(tasks) > 0:
            print('Downloading from ee...')
            print('This may take a while (up to 40 minutes).')
            print('This requires a Google Earth Engine account and at least 30MB of available storage space.')
            print('Please note that this download step is only required once.')

            self._wait_for_tasks(tasks)

            for country in ds.countries:
                df = self._load_df_country(country, filename, bands, cache)
                dfs.append(df)

        df = pd.concat(dfs)
        return df

    def _load_from_ee(self, ds, bands, passed, future, cache=True):
        """
        This function is called if the data is not available locally.
        The function willl try to load the data from GEE.
        If not available on GEE, it will not launch any download task, and return None.
        """
        dfs = []
        for country in ds.countries:
            start_date, end_date = self._get_offset_dates(
                country.start_date, country.end_date,
                passed, future
            )
            filename = self._get_filename(country, start_date, end_date, bands)

            try:
                df = self._load_df_country(country, filename, bands, cache)
                dfs.append(df)
                continue

            except ee.EEException as e:
                print(e)
                print(f'File for {country.name} does not exist. Please download it first.')
                return None

        df = pd.concat(dfs)
        return df

    def _load_locally(self, ds, bands, passed, future):
        """
        Try to load the data from local files.
        If files are not available locally, return None.
        """
        dfs = []
        for country in ds.countries:
            start_date, end_date = self._get_offset_dates(
                country.start_date, country.end_date,
                passed, future
            )
            filename = self._get_filename(country, start_date, end_date, bands)

            try:
                df = pd.read_csv(self.cache_folder + filename + '.csv')
                dfs.append(df)

            except FileNotFoundError:
                print(f'File for {country.name} does not exist. Please download it first.')
                return None

        df = pd.concat(dfs)
        return df

    def _load_df_country(self, country, filename, bands, cache):
        stats_fcc = ee.FeatureCollection(self.basename + filename)
        stats_dict = fc_to_dict(stats_fcc).getInfo()

        df = self._prepare_df(country, stats_dict, bands)

        if cache:
            self._cache(filename, df)

        print(f'File for {country.name} already exists. Nothing more to download.')
        return df

    def _get_offset_dates(self, start_date, end_date, passed, future):
        start_date = (pd.to_datetime(start_date) - passed).strftime('%Y-%m-%d')
        end_date = (pd.to_datetime(end_date) + future).strftime('%Y-%m-%d')
        return start_date, end_date

    def _get_filename(self, country, start_date, end_date, bands):
        return (f'{self.collection_name.replace("/", "_")}_'
            + f'{country.name.replace(" ", "_")}_'
            + f'{str(start_date)}_{str(end_date)}'
        )

    def _cache(self, filename, df):
        df.to_csv(self.cache_folder + filename + '.csv', index=False)

    def _wait_for_task(self, task, country):
        start = time.time()
        while task.status()['state'] != 'COMPLETED':
            elapsed = time.time() - start
            elapsed = f'{elapsed // 60:.0f}m {elapsed % 60:.0f}s'
            print(f'Current status for {country}: {task.status()["state"]}, {elapsed} elapsed')
            time.sleep(30)

    def _wait_for_tasks(self, tasks):
        for (task, country) in tasks:
            self._wait_for_task(task, country)