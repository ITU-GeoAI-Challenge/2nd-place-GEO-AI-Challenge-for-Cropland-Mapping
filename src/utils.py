import ee
import folium

import numpy as np
import pandas as pd

from src.constants import LON, LAT, TIMESTAMP, ID, COUNTRY, RADIUS_CIRCLES, C_CRICLES, WEIGHT_CIRCLES, B8, B4, NDVI

############################### 
##### Filtering functions #####
###############################

def filter_by_bounds(points, bounds):
    """
    Return a boolean mask of points that are inside the bounds (including the bounds themselves).
    """
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
    """
    Return a boolean mask of points that are in [start_date, end_date]
    """
    return points.loc[
        (points[TIMESTAMP] >= start_date) &
        (points[TIMESTAMP] <= end_date)
    ]

def filter_by_country(df, country):
    """
    Return a boolean mask of points that are in the country.
    """
    if country is None:
        return np.ones(df.shape[0]).astype('bool')
    return df[COUNTRY] == country.name

###################################
##### Interpolation functions #####
###################################

def interpolate_ts(df, bands, sampling_rate=5, start_date=None, end_date=None):
    """
    Reindex, interpolate and resample the bands of time series in df.
    The resulting time series will be aligned, without missing values, and of same lengths.
    """
    df = df.sort_values(by=[ID, TIMESTAMP])

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

def interpolate_nans_3D(X):
    """
    Given a 3D numpy array of shape (n_objects, n_timesteps, n_bands),
    interpolate missing time steps for each object.
    """
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

###########################
##### Misc. functions #####
###########################

def save_submission(preds, ids, filename):
    """
    Save the predictions in a csv file for submission.
    The file is saved in the submissions folder and prefixed with the current date and time.
    """
    df = pd.DataFrame({'ID': ids, 'Target': preds})
    
    folder = 'submissions'
    time = pd.Timestamp.now().strftime('%m_%d_%Hh_%Mm_%Ss')
    filename = f'{folder}/{time}_{filename}.csv'
    
    df.to_csv(filename, index = False)

def get_bounds(points):
    """
    Return the bounds of a set of points.
    """
    min_lon = points[LON].min()
    max_lon = points[LON].max()
    min_lat = points[LAT].min()
    max_lat = points[LAT].max()

    return [[min_lat, min_lon], [max_lat, max_lon]]

def get_center(points):
    """
    Return the center position of a set of points.
    """
    min_, max_ = get_bounds(points)
    min_lat, min_lon = min_
    max_lat, max_lon = max_

    return [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]

#########################
##### GEE functions #####
#########################

def fc_from_points(points):
    """
    Return a ee.FeatureCollection of the given points.
    """
    return ee.FeatureCollection([
        ee.Geometry.Point(lon, lat) for lon, lat in zip(points[LON], points[LAT])
    ])

def fc_to_dict(fc):
    """
    Transfer a ee.FeatureCollection to a ee.Dictionary.
    """
    prop_names = fc.first().propertyNames()
    prop_lists = fc.reduceColumns(
        reducer=ee.Reducer.toList().repeat(prop_names.size()),
        selectors=prop_names).get('list')
    return ee.Dictionary.fromLists(prop_names, prop_lists)

def addNDVI(image):
    """
    Return a ee.Image with NDVI band added. Input image must be a Sentinel-2 ee.Image.
    """
    ndvi = image.normalizedDifference([B8, B4]).rename(NDVI)
    return image.addBands([ndvi])

###################################
##### Visualization functions #####
###################################

def base_folium(ds):
    """
    Return a basic folium map centered on the center of the dataset with circle markers around the countries.
    """
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

def str_coord(coord):
    """
    Returns a string representation of a coordinate.
    """
    return f'[{coord[0]:.2f}, {coord[1]:.2f}]'