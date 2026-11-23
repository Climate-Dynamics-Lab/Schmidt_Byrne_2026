"""
Extract daily wave-filtered fields at each grid point's hot-day dates.

Output (lead_time, year, latitude, longitude), plus a 'wave' dimension for the wave terms:
  mvimd, mvimd_anom (anomaly from the daily climatology), vimfd, vimadv, vimdiv (= vimfd - vimadv)
"""
import xarray as xr
from wave_vimd_utils import extract_events

VIMD_DIR = '/home/users/lucalino/data/era5_wave_data/vimd_data/'
OUT_FILE = '/home/users/lucalino/data/era5_wave_data/vimd_data/vimd_wave_hotdays.nc'
DHD_FILE = '/gws/ssde/j25b/global_ex/lms47/data/era5_derivatives/era5_hot_day_full_col_mb_components_extreme_var=t2m_region=tropics.zarr'
LAT_KEEP = 20.                                                  # drop the ±21 rows (one-sided derivatives)
SMOOTHING_WINDOW = 21                                           # days; running mean of the daily climatology

def smooth_cyclic(clim, window, dim='dayofyear'):
    """Centred running mean over `window` days, wrapping around the end of the year."""
    n = window // 2
    extended = xr.concat([clim.isel({dim: slice(-n, None)}), clim, clim.isel({dim: slice(0, n)})], dim=dim)
    smoothed = extended.rolling({dim: window}, center=True).mean().isel({dim: slice(n, -n)})
    return smoothed.assign_coords({dim: clim[dim]})

def climatological_anomaly(da, smoothing_window=SMOOTHING_WINDOW):
    """Anomaly from the 1980-2018 mean of each calendar day, smoothed with a cyclic running mean."""
    clim = da.groupby('time.dayofyear').mean('time')
    if smoothing_window:
        clim = smooth_cyclic(clim, smoothing_window)
    return (da.groupby('time.dayofyear') - clim).drop_vars('dayofyear')

def extract_all(dwave, dhd_time):
    """daily: Dataset (wave, time, lat, lon); dhd_time: datetimes (lead_time, year, lat, lon) on 0.25°."""
    # drop the ±21 rows first (no hot-day dates there)
    dwave = dwave.sel(latitude=abs(dwave.latitude) <= LAT_KEEP)
    # Design choice: hot-day dates of the 0.25° point at each 1° cell centre
    times = dhd_time.sel(latitude=dwave.latitude, longitude=dwave.longitude, method='nearest', tolerance=1e-3)
    times = times.assign_coords(latitude=dwave.latitude, longitude=dwave.longitude).dt.floor('D')
    # only the years covered by the dwave files (dates of edge events outside them stay NaN)
    years = dwave.time.dt.year
    times = times.sel(year=slice(int(years.min()), int(years.max())))

    mvimd = dwave.mvimd.load()
    out = {
        'mvimd': extract_events(mvimd, times),
        'mvimd_anom': extract_events(climatological_anomaly(mvimd), times),
    }
    for name in ['vimfd', 'vimadv']:
        out[name] = xr.concat([extract_events(dwave[name].sel(wave=w).load(), times) for w in dwave.wave.values],
                              dim=dwave.wave)
    out['vimdiv'] = out['vimfd'] - out['vimadv']

    ds = xr.Dataset(out)
    for name in ds.data_vars:
        ds[name].attrs = {'units': 'kg m**-2 s**-1'}
    return ds


if __name__ == '__main__':
    dwave = xr.open_mfdataset(f'{VIMD_DIR}vimd_wave_daily_*.nc', combine='by_coords').sel(latitude=slice(20,-20))
    dhd_time = xr.open_zarr(DHD_FILE, chunks={}, consolidated=True)['time']
    ds = extract_all(dwave, dhd_time)
    ds.to_netcdf(OUT_FILE, encoding={v: {'dtype': 'float32', 'zlib': True, 'complevel': 1} for v in ds.data_vars})
    print(f'{OUT_FILE} saved.')
