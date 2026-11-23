import numpy as np
import xarray as xr
import numcodecs
from utils import load_era5

# Select the hottest day per location within each year (based on extreme_var), and extract
# the temperature value on that day (lead_time=0) and the preceding day(s) up to nb_lead_days-1.
# Resulting dataset dims: (year, lead_time, latitude, longitude)

NB_LEAD_DAYS = 21
NB_LAG_DAYS = 11

lead_time_chunking = NB_LEAD_DAYS + NB_LAG_DAYS - 1
tropics_lat = slice(20, -20)
derivatives_folder = "/gws/ssde/j25b/global_ex/lms47/data/era5_derivatives/"
zarr_store = derivatives_folder + "era5_hot_day_t2m_lead" + str(NB_LEAD_DAYS-1) + "_lag" + str(NB_LAG_DAYS-1) + "_1980-2024.zarr"

# load temperature data
ds = load_era5()[["t2m"]].sel(latitude = tropics_lat)

# for each year, starting in 1980: find event time (hottest day) per grid cell and extract lead_day data
years = np.unique(ds['time.year'].values).astype(int)

# loop extracting hot days starting with 1980 and ending with 2024 
# (lag days falling into 2025 are not available, so year 2024 gets removed later in another script)
for y in years[(years >= 1980) & (years <= 2024)]: 
    # select data for this year
    ds_y = ds.sel(time=ds['time.year'] == y)
    if ds_y['time'].size == 0:
        raise ValueError(f"No data for year {y}")

    # find the corresponding time stamp for extreme day for each grid cell
    # ensure tmax is computed (no dask chunks) so it can be used as an indexer
    tmax = ds_y['t2m'].idxmax(dim='time').compute()   # dims: (latitude, longitude), dtype datetime64
    
    # build per-offset selections (lead and lag combined)
    per_lead = []
    per_lead_timestamps = []
    for L in range(-(NB_LEAD_DAYS - 1), NB_LAG_DAYS): # range symmetric around hot day (lead=0) if same nb of lead and lag days
        sel_time = tmax + np.timedelta64(L, 'D')  # dims: (latitude, longitude)
        selected = ds.sel(time=sel_time, method='nearest')
        selected = selected.expand_dims({'lead_time': [L]})
        per_lead.append(selected)
        per_lead_timestamps.append(sel_time.expand_dims({'lead_time': [L]}))
    
    per_year = xr.concat(per_lead, dim='lead_time')  # dims: (lead_time, latitude, longitude)
    per_year_timestamps = xr.concat(per_lead_timestamps, dim='lead_time')  # dims: (lead_time, latitude, longitude)
    
    # Sort by lead_time to get [-NB_LEAD_DAYS+1, ..., 0, ..., NB_LAG_DAYS-1]
    per_year = per_year.sortby('lead_time')
    per_year_timestamps = per_year_timestamps.sortby('lead_time')
    
    per_year = per_year.expand_dims({'year': [int(y)]})  # add year dim
    per_year_timestamps = per_year_timestamps.expand_dims({'year': [int(y)]})  # add year dim (now has shape: year, lead_time, latitude, longitude)
    per_year = per_year.assign_coords(time=per_year_timestamps)
    
    # chunk per_year data and save to zarr store
    encoding = {v: {'compressor': numcodecs.Blosc("zstd", clevel=6)} for v in per_year.data_vars}
    per_year = per_year.chunk({'year':1, 'lead_time': lead_time_chunking, 'latitude':len(per_year.latitude), 'longitude':len(per_year.longitude)})
    if y == 1980:
        per_year.to_zarr(zarr_store, mode='w', zarr_format=2, encoding=encoding, compute=True)
    else:
        per_year.to_zarr(zarr_store, mode='a', append_dim='year', compute=True)


