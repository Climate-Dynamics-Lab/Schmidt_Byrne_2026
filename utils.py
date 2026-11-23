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

def climatology_doy(da, time_dim_returned='time', mode='detrended'): # mode = 'detrended' or 'simple_mean'
    """
    Compute the climatology for each location and day of the year by fitting a linear trend to the full time series, 
    and evaluating this trend at the respective day of the first year.
    Parameters:
    da (xarray.DataArray): Input DataArray with a time dimension.
    Returns:
    xarray.Dataset: Dataset containing one variable, named '<input_var>_clim' which contains the climatology of each day and location, repeated for all years.
    """
    grouped = da.groupby("time.dayofyear")
    # subtract 1 for consistency with dayofyear starting at 0 and ending at 365
    time_doy_values = da.time.dt.dayofyear.values
    if mode == 'detrended':
        clim_list = []
        time_dates = da.time

        for day, group in grouped:
            # fit linear trend
            coefs = group.polyfit(dim="time", deg=1, skipna=True)

            # evaluate the fit at each point in time
            trend = xr.polyval(group["time"], coefs)

            # extract climatology as the baseline value of the trend (use first time of the group)
            t0 = group["time"].isel(time=0)
            clim = trend.sel(time=t0)
            # attach a dayofyear coordinate so concat produces a dayofyear axis (0-based)
            clim = clim.expand_dims(dayofyear=[int(day)])
            clim_list.append(clim)

        # Concatenate climatology entries into a single DataArray indexed by dayofyear
        climatology = xr.concat(clim_list, dim="dayofyear")
        del climatology.coords["time"]
        var_name = "polyfit_coefficients"
    elif mode == 'simple_mean':
        climatology = grouped.mean()
    if time_dim_returned == 'time':
        # select the climatology value for each original time using the dayofyear values
        climatology = climatology.sel(dayofyear=time_doy_values)
        # rename dimension dayofyear to time and assign original time coordinates
        climatology = climatology.rename({"dayofyear": "time"})
        climatology = climatology.assign_coords(time=time_dates)
    elif time_dim_returned == 'doy':
        pass  # keep dayofyear dimension
    # name climatology variable and attach attributes
    var_name_clim = da.name + '_clim'
    long_name_string = f' (day-of-year climatology, {mode})'
    if mode == 'detrended':
        climatology = climatology.rename_vars({var_name:var_name_clim})
        climatology[var_name_clim].attrs = da.attrs.copy()        
        climatology[var_name_clim].attrs['long_name'] = climatology[var_name_clim].attrs['long_name'] + long_name_string
        return climatology[var_name_clim]
    elif mode == 'simple_mean':
        climatology = climatology.rename(var_name_clim)
        climatology.attrs['short_name'] = var_name_clim
        climatology.attrs['long_name'] = climatology.attrs['long_name'] + long_name_string
        return climatology

def extend_dayofyear_cyclic(ds, n):
    """
    Extend a dataset with cyclic boundary conditions on the dayofyear dimension.
    
    Prepends the last n days of the year (with adjusted dayofyear indices) and appends 
    the first n days of the year (with adjusted dayofyear indices). This is useful for 
    preparing data for cyclic rolling operations.
    
    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Input data with a 'dayofyear' dimension. Typically dayofyear ranges from 0 to 365.
    n : int
        Number of days to add at beginning and end
        
    Returns
    -------
    xr.Dataset or xr.DataArray
        Extended dataset with dayofyear dimension now ranging from (-n) to (365+n)
        depending on original size.
        
    Examples
    --------
    >>> # Original dayofyear: 0-365
    >>> ds_extended = extend_dayofyear_cyclic(ds, n=15)
    >>> # New dayofyear: -15 to 380
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
