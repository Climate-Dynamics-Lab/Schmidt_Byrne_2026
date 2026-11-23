import sys
sys.path.append('/home/users/lucalino/')
import numpy as np
import xarray as xr
import warnings
from dask.array.core import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)

######### SETTINGS ########        
epsilon = 0.0
lead_time_window = slice(-4,-1) # time window over which to calculate mean drying/moistening
###########################

secperday = 24*60*60
land_thres = 0.5
derivatives_folder = "/gws/ssde/j25b/global_ex/lms47/data/era5_derivatives/"
composite_folder = "t2m_composites/met_vars/" 

# load data
dmet = xr.open_zarr(derivatives_folder + f'era5_hot_day_met_vars_extreme_var=t2m_region=tropics.zarr', chunks={}, consolidated=True)
vars_to_nc = [var for var in dmet.data_vars]
dmet['lsm'] = xr.open_dataset(derivatives_folder + 'era5_land_sea_mask_region=tropics.nc')['lsm']
dmet['ai_percentile'] = xr.open_dataarray(derivatives_folder + 'era5_ai_percentiles_region=tropics.nc')
dmet['d_tcwv_dt'] = xr.open_zarr(derivatives_folder + f'era5_hot_day_full_col_mb_components_extreme_var=t2m_region=tropics.zarr', chunks={}, consolidated=True)['d_tcwv_dt']

# aridity categories
arid = (dmet.ai_percentile >= 0) & (dmet.ai_percentile < 25)
semi_arid = (dmet.ai_percentile >= 25) & (dmet.ai_percentile < 50)
semi_moist = (dmet.ai_percentile >= 50) & (dmet.ai_percentile < 75)
moist = (dmet.ai_percentile >= 75) & (dmet.ai_percentile <= 100)
land = dmet.lsm > land_thres
ocean = dmet.lsm <= land_thres

# mean drying versus moistening before hot day
neg_dq = (dmet['d_tcwv_dt'].sel(lead_time=lead_time_window).mean('lead_time') < -epsilon).compute()
pos_dq = (dmet['d_tcwv_dt'].sel(lead_time=lead_time_window).mean('lead_time') > epsilon).compute()
no_dq = (abs(dmet['d_tcwv_dt'].sel(lead_time=lead_time_window).mean('lead_time')) <= epsilon).compute()

# calculate area-weighted data for composite plots and store as netCDF files
weights = np.cos(np.deg2rad(dmet.latitude))

dmoist_drying = dmet[vars_to_nc].where(moist & land & neg_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
dmoist_drying.to_netcdf(derivatives_folder + composite_folder + 'met_vars_dmoist_drying_extreme_var=t2m_dqcrit=tcwv_epsilon=' + str(epsilon) + '.nc')
dsemimoist_drying = dmet[vars_to_nc].where(semi_moist & land & neg_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
dsemimoist_drying.to_netcdf(derivatives_folder + composite_folder + 'met_vars_dsemimoist_drying_extreme_var=t2m_dqcrit=tcwv_epsilon=' + str(epsilon) + '.nc')
dsemiarid_drying = dmet[vars_to_nc].where(semi_arid & land & neg_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
dsemiarid_drying.to_netcdf(derivatives_folder + composite_folder + 'met_vars_dsemiarid_drying_extreme_var=t2m_dqcrit=tcwv_epsilon=' + str(epsilon) + '.nc')
darid_drying = dmet[vars_to_nc].where(arid & land & neg_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
darid_drying.to_netcdf(derivatives_folder + composite_folder + 'met_vars_darid_drying_extreme_var=t2m_dqcrit=tcwv_epsilon=' + str(epsilon) + '.nc')

dmoist_moistening = dmet[vars_to_nc].where(moist & land & pos_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
dmoist_moistening.to_netcdf(derivatives_folder + composite_folder + 'met_vars_dmoist_moistening_extreme_var=t2m_dqcrit=tcwv_epsilon=' + str(epsilon) + '.nc')
dsemimoist_moistening = dmet[vars_to_nc].where(semi_moist & land & pos_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
dsemimoist_moistening.to_netcdf(derivatives_folder + composite_folder + 'met_vars_dsemimoist_moistening_extreme_var=t2m_dqcrit=tcwv_epsilon=' + str(epsilon) + '.nc')
dsemiarid_moistening = dmet[vars_to_nc].where(semi_arid & land & pos_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
dsemiarid_moistening.to_netcdf(derivatives_folder + composite_folder + 'met_vars_dsemiarid_moistening_extreme_var=t2m_dqcrit=tcwv_epsilon=' + str(epsilon) + '.nc')
darid_moistening = dmet[vars_to_nc].where(arid & land & pos_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
darid_moistening.to_netcdf(derivatives_folder + composite_folder + 'met_vars_darid_moistening_extreme_var=t2m_dqcrit=tcwv_epsilon=' + str(epsilon) + '.nc')