import numpy as np
import xarray as xr
import numcodecs
import scipy.stats as stats

def aridity_index(nlw, nsw, p):
    # AI = P / PET = Lv * P / (0.8 * Rnet)
    ai = (physical_constants()['L_v'] * p) / (0.8 * (nsw + nlw))
    ai.attrs = {'short_name': 'ai',
                'long_name': 'Aridity index',
                'units': '',
                'description': 'AI = P/PET, with PET definition from Milly & Dunne (2016)',
                }
    return ai

def aridity_index_percentiles(ai_land):
    # Rank all land values
    land_mask = ~np.isnan(ai_land.values)
    ai_land_values = ai_land.values[land_mask]
    ranks = stats.rankdata(ai_land_values, nan_policy='omit')
    percentiles_1d = 100 * (ranks - 1) / (len(ranks) - 1)
    # Map back to 2D grid
    percentiles_2d = np.full_like(ai_land.values, np.nan)
    percentiles_2d[land_mask] = percentiles_1d
    ai_percentile = ai_land.copy()
    ai_percentile.values = percentiles_2d
    ai_percentile.attrs['long_name'] = 'Aridity index percentile'
    ai_percentile.attrs['units'] = '%'
    return ai_percentile

def climatology_doy(da, mode='simple_mean'):
    grouped = da.groupby("time.dayofyear")
    climatology = grouped.mean()
    var_name_clim = da.name + '_clim'
    long_name_string = f' (day-of-year climatology, {mode})'
    climatology = climatology.rename(var_name_clim)
    climatology.attrs['short_name'] = var_name_clim
    climatology.attrs['long_name'] = climatology.attrs['long_name'] + long_name_string
    return climatology

def extend_dayofyear_cyclic(ds, n):
    """
    Extend a dataset with cyclic boundary conditions on the dayofyear dimension.
    
    Prepends the last n days of the year (with adjusted dayofyear indices) and appends 
    the first n days of the year (with adjusted dayofyear indices). This is needed for 
    preparing data for cyclic rolling operations.
    """
      
    # Get original dayofyear values and size
    doy_original = ds.dayofyear.values
    doy_size = len(doy_original)
    
    # Create extended dayofyear coordinate
    # Last n days (from end): doy_size - n to doy_size - 1
    doy_end_indices = np.arange(doy_size - n, doy_size)
    # First n days: 0 to n - 1
    doy_start_indices = np.arange(0, n)
    
    # Get the original dayofyear values
    doy_end_values = doy_original[doy_end_indices]
    doy_start_values = doy_original[doy_start_indices]
    
    # Adjust coordinates: wrap around cyclically
    # For the end values prepended: subtract doy_size from them
    doy_prepend = doy_end_values - doy_size
    # For the start values appended: add doy_size to them
    doy_append = doy_start_values + doy_size
    
    # # For Dataset, extend each data variable
    # ds_extended = ds.copy()
    
    # Extract the parts to prepend and append
    ds_end = ds.isel(dayofyear=doy_end_indices)
    ds_start = ds.isel(dayofyear=doy_start_indices)
    
    # Assign new dayofyear coordinates
    ds_end['dayofyear'] = doy_prepend
    ds_start['dayofyear'] = doy_append
    
    # Concatenate
    ds_extended = xr.concat([ds_end, ds, ds_start], dim='dayofyear')    
    return ds_extended

def get_compressor():
    return numcodecs.Blosc("zstd", clevel=6) # shuffle = 2 for bitrounding (after rounding was performed for variables separately)

def load_era5(version = "daily"):
    path = "/gws/ssde/j25b/global_ex/data/"
    if version == "daily":
        file_name = "era5.zarr"
    elif version == "monthly":
        file_name = "era5_monthly.zarr"
    d = xr.open_zarr(
        path + file_name,
        chunks={}, 
        consolidated=True,
    )
    return d

def physical_constants():
    pc = {
        'epsilon': 0.622, # ratio of molecular weight of water vapor and dry air
        'g0': 9.80665,  # gravitational acceleration constant in m/s²
        'r_e': 6.371e6,  # Earth radius [m]
        'L_v': 2.5008e6,  # Latent heat of vaporization [J/kg] = [m2/s2]
        'kappa': 0.286, # R/c_p (specific gas constant of air divided by specific heat capacity at constant pressure)
        'c_p': 1005.0, # Specific heat capacity of air [J/(kg*K)]
    }
    return pc

def temporal_smoothing(ds, excluded_vars=[], window_size=3, center=True, min_periods=1):
    d = ds.copy()
    float_vars = [
        var for var, da in d.data_vars.items()
        if np.issubdtype(da.dtype, np.floating) and var not in excluded_vars
    ]
    dsel = d[float_vars]
    if 'lead_time' in d.dims:
        ds_smoothed = dsel.rolling(lead_time=window_size, center=center, min_periods=min_periods).mean()
    elif 'dayofyear' in d.dims:
        ds_smoothed = dsel.rolling(dayofyear=window_size, center=center, min_periods=min_periods).mean()
    else:
        ds_smoothed = dsel.rolling(time=window_size, center=center, min_periods=min_periods).mean()
    for var in float_vars:
        if 'description' in d[var].attrs:
            ds_smoothed[var].attrs['description'] = d[var].attrs['description'] + ',\nvalues temporally smoothed by averaging over rolling window of ' + str(window_size) + ' time steps'
        else:
            ds_smoothed[var].attrs['description'] = 'Values temporally smoothed by averaging over rolling window of ' + str(window_size) + ' time steps'
        d[var] = ds_smoothed[var]
    return d

def time_derivative(da, coord = 'time', datetime_unit = "D"):
    d_da_dt_name = "d_" + da.short_name + "_dt"
    if coord == 'time':
        d_da_dt = da.chunk(time = 20).differentiate(coord, datetime_unit = datetime_unit).chunk(time = 10)
    elif coord == 'lead_time':
        d_da_dt = da.chunk(lead_time = -1).differentiate(coord, datetime_unit = None).chunk(lead_time = 10)
    d_da_dt.attrs = {'short_name': d_da_dt_name,
                     'long_name': f'Time-derivative of {da.short_name}', 
                     'units': f'{da.attrs["units"]} {datetime_unit.lower()}-1'}
    return d_da_dt
