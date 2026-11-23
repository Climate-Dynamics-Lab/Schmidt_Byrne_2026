import numpy as np
import xarray as xr
from utils import time_derivative, load_era5, climatology_doy, extend_dayofyear_cyclic, temporal_smoothing
import math

var = 'd_tcwv_dt' # set variable to compute climatology for
time_dim_to_store = 'doy' # options: 'time' or 'doy'
smoothing_window = 21
tropics_lat = slice(20, -20)
smoothing_string = 'from_smoothed_timeseries_ws=' + str(smoothing_window)
derivatives_folder = "/gws/ssde/j25b/global_ex/lms47/data/era5_derivatives/doy_climatologies/"
out_filename = f'era5_doy_climatology_var={var}_region=tropics_timedim=doy_mode=simple_mean_{smoothing_string}.nc'

if var == 'd_tcwv_dt':
    ds = load_era5()[['tcwv']].sel(latitude = tropics_lat)
    ds['d_tcwv_dt'] = time_derivative(ds['tcwv']).compute()
else:
    ds = load_era5()[[var]].sel(latitude = tropics_lat)

# add climatological mean value for each day of year
ds[f'{var}_clim'] = climatology_doy(ds[var], time_dim_returned='doy', mode='simple_mean')

# smooth the day-of-year climatology with a running window of size smoothing_window (in days)
ds_clim = extend_dayofyear_cyclic(ds[[f'{var}_clim']], n=math.ceil(smoothing_window/2))
ds_clim = temporal_smoothing(ds_clim, window_size=smoothing_window)

# save smoothed day-of-year climatology of var to netcdf
ds_clim[f'{var}_clim'].sel(dayofyear=slice(1, 366)).to_netcdf(derivatives_folder + out_filename)