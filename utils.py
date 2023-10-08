import ee
import folium

import numpy as np
import pandas as pd

from constants import LON, LAT, TIMESTAMP, ID, COUNTRY, RADIUS_CIRCLES, C_CRICLES, WEIGHT_CIRCLES

def get_bounds(points):
    min_lon = points[LON].min()
    max_lon = points[LON].max()
    min_lat = points[LAT].min()
    max_lat = points[LAT].max()

    return [[min_lat, min_lon], [max_lat, max_lon]]

def get_center(points):
    min_, max_ = get_bounds(points)
    min_lat, min_lon = min_
    max_lat, max_lon = max_

    return [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]

def str_coord(coord):
    return f'[{coord[0]:.2f}, {coord[1]:.2f}]'

def filter_by_bounds(points, bounds):
    min_, max_ = bounds
    min_lat, min_lon = min_
    max_lat, max_lon = max_

    return (
        (points[LAT] >= min_lat) &
        (points[LAT] <= max_lat) &
        (points[LON] >= min_lon) &
        (points[LON] <= max_lon)
    )

def filter_by_dates(points, start_date, end_date):
    return points.loc[
        (points[TIMESTAMP] >= start_date) &
        (points[TIMESTAMP] <= end_date)
    ]

def filter_by_country(df, country):
    if country is None:
        return np.ones(df.shape[0]).astype('bool')
    return df[COUNTRY] == country.name

def fc_from_points(points):
    return ee.FeatureCollection([
        ee.Geometry.Point(lon, lat) for lon, lat in zip(points[LON], points[LAT])
    ])

def base_folium(ds):
    m = folium.Map(location=ds.center, zoom_start=5)
    
    folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = False
    ).add_to(m)
    folium.TileLayer(
        tiles = 'Stamen Toner',
        attr = 'Stamen',
        name = 'Stamen Toner',
        overlay = True,
        control = True,
        opacity = 0.4
    ).add_to(m)
    
    fg = folium.FeatureGroup(name='Countries')
    for country in ds.countries:
        circle = folium.Circle(
            location=country.center, 
            radius=RADIUS_CIRCLES, color=C_CRICLES, weight=WEIGHT_CIRCLES)
        fg.add_child(circle)
    m.add_child(fg)

    return m

def save_submission(preds, ids, filename):
    df = pd.DataFrame({'ID': ids, 'Target': preds})
    time = pd.Timestamp.now().strftime('%m_%d_%Hh_%Mm_%Ss')
    folder = 'submissions'
    
    filename = f'{folder}/{time}_{filename}.csv'
    
    df.to_csv(filename, index = False)

def addNDVI(image):
    ndvi = image.normalizedDifference([B8, B4]).rename(NDVI)
    return image.addBands([ndvi])

def interpolate_nans_3D(X):
    n_objects, n_timesteps, n_bands = X.shape
    isnan = np.isnan(X[:, :, 0])
    for i, x in enumerate(X):
        valid_timesteps = np.arange(n_timesteps)[~isnan[i]]
        valid_values = X[i, valid_timesteps]

        # Perform linear interpolation and extrapolation
        for j in range(n_bands):
            X[i, :, j] = np.interp(
                np.arange(n_timesteps), 
                valid_timesteps,
                valid_values[:, j]
            )
    return X

def interpolate_ts(df, bands, sampling_rate=5, start_date=None, end_date=None):
    df = df.sort_values(by=[ID, TIMESTAMP])

    # bands_train is a list of timeseries of shape (n_timesteps, n_bands)
    # but each timeserie can have a different number of timestamps so we will interpolate 
    # the missing values between start and end date

    if start_date is None:
        start_date = pd.to_datetime(df[TIMESTAMP].min())
    if end_date is None:
        end_date = pd.to_datetime(df[TIMESTAMP].max())
    date_range = pd.date_range(start_date, end_date, freq='D', normalize=True)

    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP]).dt.normalize()
    df = df.drop_duplicates(subset=[ID, TIMESTAMP])

    groups = df.groupby(ID)
    XX = groups.apply(
        lambda x: x.set_index(TIMESTAMP)[bands].reindex(date_range).values
    )
    IDs = groups[ID].first().values

    XX = np.stack(XX.values).astype('float32')
    XX = interpolate_nans_3D(XX)
    XX = XX[:, ::sampling_rate, :]

    return XX, IDs

# Function to transfer feature properties to a dictionary.
def fc_to_dict(fc):
  prop_names = fc.first().propertyNames()
  prop_lists = fc.reduceColumns(
      reducer=ee.Reducer.toList().repeat(prop_names.size()),
      selectors=prop_names).get('list')

  return ee.Dictionary.fromLists(prop_names, prop_lists)
