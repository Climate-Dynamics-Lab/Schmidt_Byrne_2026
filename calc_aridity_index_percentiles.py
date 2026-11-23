import numpy as np
import xarray as xr
from utils import load_era5, aridity_index, aridity_index_percentiles

land_thres = 0.5
tropics_lat = slice(20, -20)
derivatives_folder = "/gws/nopw/j04/global_ex/lms47/data/era5_derivatives/"
out_filename = 'era5_ai_percentiles_region=tropics.nc'

ds = load_era5()[['mtpr', 'msnlwrf', 'msnswrf', 'lsm']].sel(latitude = tropics_lat)
ai = aridity_index(ds.msnlwrf, ds.msnswrf, ds.mtpr)
ai_land = ai.where(ds.lsm.isel(time = 0) > land_thres)
ai_percentile = aridity_index_percentiles(ai_land)
ai_percentile = ai_percentile.rename_vars({'__xarray_dataarray_variable__': 'ai_perc'})

# save aridity index percentile dataarray to netcdf
ai_percentile.to_netcdf(derivatives_folder + out_filename)