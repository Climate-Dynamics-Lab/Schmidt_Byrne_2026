import numpy as np
import xarray as xr
from utils import time_derivative, load_era5, climatology_doy, extend_dayofyear_cyclic, temporal_smoothing
import math

VAR = 'd_tcwv_dt' # set variable to compute climatology for
SMOOTHING_WINDOW = 21

smoothing_string = 'from_smoothed_timeseries_ws=' + str(SMOOTHING_WINDOW)
derivatives_folder = "/gws/ssde/j25b/global_ex/lms47/data/era5_derivatives/doy_climatologies/"
out_filename = f'era5_doy_climatology_var={VAR}_region=tropics_timedim=doy_mode=simple_mean_{smoothing_string}.nc'
tropics_lat = slice(20, -20)

if VAR == 'd_tcwv_dt':
    ds = load_era5()[['tcwv']].sel(latitude = tropics_lat)
    ds['d_tcwv_dt'] = time_derivative(ds['tcwv']).compute()
else:
    ds = load_era5()[[VAR]].sel(latitude = tropics_lat)

# add climatological mean value for each day of year
ds[f'{VAR}_clim'] = climatology_doy(ds[VAR], mode='simple_mean')

# smooth the day-of-year climatology with a running window of size smoothing_window (in days)
ds_clim = extend_dayofyear_cyclic(ds[[f'{VAR}_clim']], n=math.ceil(SMOOTHING_WINDOW/2))
ds_clim = temporal_smoothing(ds_clim, window_size=SMOOTHING_WINDOW)

# save smoothed day-of-year climatology of VAR to netcdf
ds_clim[f'{VAR}_clim'].sel(dayofyear=slice(1, 366)).to_netcdf(derivatives_folder + out_filename)