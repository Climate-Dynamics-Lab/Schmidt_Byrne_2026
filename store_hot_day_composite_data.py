import numpy as np
import xarray as xr
import warnings
from dask.array.core import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)


####### SETTINGS ########
doy_clim_comparison = True
lead_time_window = slice(-4,-1) # time window over which to calculate mean drying/moistening
epsilon = 0.0
moistening_drying = True
#########################

def add_doy_clim_variables(ds, vars=['tcwv', 'd_tcwv_dt', 't2m', 'q2m', 'mer', 'mvimd', 'mtpr']):
    for var in vars:
        ds_clim = xr.open_dataset(f'/gws/ssde/j25b/global_ex/lms47/data/era5_derivatives/doy_climatologies/era5_doy_climatology_var={var}_region=tropics_timedim=doy_mode=simple_mean_from_smoothed_timeseries_ws=21.nc')
        ds[var+'_clim'] = ds_clim[var+'_clim'].sel(dayofyear = ds.time.dt.dayofyear)
    return ds

secperday = 24*60*60
land_thres = 0.5
derivatives_folder = "/gws/ssde/j25b/global_ex/lms47/data/era5_derivatives/"
composite_folder = "t2m_composites/"

# load hot day moisture budget components
dex = xr.open_zarr(derivatives_folder + f'era5_hot_day_full_col_mb_components_extreme_var=t2m_region=tropics.zarr', chunks={}, consolidated=True)
dex = dex.assign(qmvimd = dex.mvimd - dex.viiw - dex.vilw)
dex['qmvimd'].attrs = dex['mvimd'].attrs
dex['qmvimd'].attrs['description'] = 'water vapour component, computed as a residual from mvimd - viiw - vilw'
vars_to_nc = ['t2m', 'tcwv', 'd_tcwv_dt', 'q2m', 'mer', 'mvimd', 'mtpr', 'qmvimd']

if doy_clim_comparison:
    dex = add_doy_clim_variables(dex)
    vars_to_nc = ['t2m_diff_clim_abs', 'tcwv_diff_clim_abs', 'q2m_diff_clim_abs', 'mer_diff_clim_abs', 
                  'mvimd_diff_clim_abs', 'mtpr_diff_clim_abs', 'd_tcwv_dt_diff_clim_abs']
    for var in ['t2m', 'tcwv', 'd_tcwv_dt','q2m', 'mer', 'mvimd', 'mtpr']:
        dex[var+'_diff_clim_abs'] = dex[var] - dex[var+'_clim']
    clim_string = '_doy_clim_comparison'
else:
    clim_string = ''

dex = dex.assign(lsm = xr.open_dataarray(derivatives_folder +'era5_land_sea_mask_region=tropics.nc'))
dex = dex.assign(ai_percentile = xr.open_dataarray(derivatives_folder + 'era5_ai_percentiles_region=tropics.nc'))

# aridity categories
arid = (dex.ai_percentile >= 0) & (dex.ai_percentile < 25)
semi_arid = (dex.ai_percentile >= 25) & (dex.ai_percentile < 50)
semi_moist = (dex.ai_percentile >= 50) & (dex.ai_percentile < 75)
moist = (dex.ai_percentile >= 75) & (dex.ai_percentile <= 100)
land = dex.lsm > land_thres
ocean = dex.lsm < land_thres

# mean drying versus moistening during lead days specified by lead_time_window 
if moistening_drying:
    neg_dq = (dex['d_tcwv_dt'].sel(lead_time=lead_time_window).mean('lead_time') < -epsilon).compute()
    pos_dq = (dex['d_tcwv_dt'].sel(lead_time=lead_time_window).mean('lead_time') > epsilon).compute()

# calculate area-weighted data for composite plots and store as netCDF files
weights = np.cos(np.deg2rad(dex.latitude))

dmoist_drying = dex[vars_to_nc].where(moist & land & neg_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
dmoist_drying.to_netcdf(derivatives_folder + composite_folder + 'full_col_dmoist_drying_dqcrit=tcwv_epsilon=' + str(epsilon) + clim_string + '.nc')
dsemimoist_drying = dex[vars_to_nc].where(semi_moist & land & neg_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
dsemimoist_drying.to_netcdf(derivatives_folder + composite_folder + 'full_col_dsemimoist_drying_dqcrit=tcwv_epsilon=' + str(epsilon) + clim_string + '.nc')
dsemiarid_drying = dex[vars_to_nc].where(semi_arid & land & neg_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
dsemiarid_drying.to_netcdf(derivatives_folder + composite_folder + 'full_col_dsemiarid_drying_dqcrit=tcwv_epsilon=' + str(epsilon) + clim_string + '.nc')
darid_drying = dex[vars_to_nc].where(arid & land & neg_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
darid_drying.to_netcdf(derivatives_folder + composite_folder + 'full_col_darid_drying_dqcrit=tcwv_epsilon=' + str(epsilon) + clim_string + '.nc')
dmoist_moistening = dex[vars_to_nc].where(moist & land & pos_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
dmoist_moistening.to_netcdf(derivatives_folder + composite_folder + 'full_col_dmoist_moistening_dqcrit=tcwv_epsilon=' + str(epsilon) + clim_string + '.nc')
dsemimoist_moistening = dex[vars_to_nc].where(semi_moist & land & pos_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
dsemimoist_moistening.to_netcdf(derivatives_folder + composite_folder + 'full_col_dsemimoist_moistening_dqcrit=tcwv_epsilon=' + str(epsilon) + clim_string + '.nc')
dsemiarid_moistening = dex[vars_to_nc].where(semi_arid & land & pos_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
dsemiarid_moistening.to_netcdf(derivatives_folder + composite_folder + 'full_col_dsemiarid_moistening_dqcrit=tcwv_epsilon=' + str(epsilon) + clim_string + '.nc')
darid_moistening = dex[vars_to_nc].where(arid & land & pos_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
darid_moistening.to_netcdf(derivatives_folder + composite_folder + 'full_col_darid_moistening_dqcrit=tcwv_epsilon=' + str(epsilon) + clim_string + '.nc')
docean_drying = dex[vars_to_nc].where(ocean & neg_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
docean_drying.to_netcdf(derivatives_folder + composite_folder + 'full_col_docean_drying_dqcrit=tcwv_epsilon=' + str(epsilon) + clim_string + '.nc')
docean_moistening = dex[vars_to_nc].where(ocean & pos_dq).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
docean_moistening.to_netcdf(derivatives_folder + composite_folder + 'full_col_docean_moistening_dqcrit=tcwv_epsilon=' + str(epsilon) + clim_string + '.nc')


# Composite data from all events used to gauge suitable lead_time_window for moistening/drying classification

# dmoist_all_events = dex[vars_to_nc].where(moist & land).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
# dmoist_all_events.to_netcdf(derivatives_folder + composite_folder + 'full_col_dmoist_all_events'+ clim_string + '.nc')
# dsemimoist_all_events = dex[vars_to_nc].where(semi_moist & land).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
# dsemimoist_all_events.to_netcdf(derivatives_folder + composite_folder + 'full_col_dsemimoist_all_events'+ clim_string + '.nc')
# dsemiarid_all_events = dex[vars_to_nc].where(semi_arid & land).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
# dsemiarid_all_events.to_netcdf(derivatives_folder + composite_folder + 'full_col_dsemiarid_all_events'+ clim_string + '.nc')
# darid_all_events = dex[vars_to_nc].where(arid & land).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
# darid_all_events.to_netcdf(derivatives_folder + composite_folder + 'full_col_darid_all_events'+ clim_string + '.nc')
# docean_all_events = dex[vars_to_nc].where(ocean).weighted(weights).mean(dim = ['year', 'latitude', 'longitude'])
# docean_all_events.to_netcdf(derivatives_folder + composite_folder + 'full_col_docean_all_events'+ clim_string + '.nc')