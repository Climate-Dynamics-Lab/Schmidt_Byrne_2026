"""
Compute daily column-integrated wave moisture flux divergence and advection for one year.

For each day of YEAR, on the 1° wave grid:
  mvimd          ERA5 mean vertically integrated moisture divergence, regridded   [kg m-2 s-1]
  vimfd(wave)    (1/g) ∫ ∇·(q v_wave) dp,  p_top .. surface                       [kg m-2 s-1]
  vimadv(wave)   (1/g) ∫ v_wave·∇q dp                                             [kg m-2 s-1]
The divergence term (1/g) ∫ q ∇·v_wave dp is vimfd - vimadv (not stored).
wave = 'era5_full' holds the same terms computed with the full (unfiltered) ERA5 winds.

Usage:  python compute_daily_vimd.py [YEAR]     (default: $SLURM_ARRAY_TASK_ID)
"""
import os
import dask
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from utils import load_era5
from wave_vimd_utils import regrid_conservative, calc_mfd_and_adv, column_integral, wave_uv_from_coefficients
dask.config.set(scheduler='threads', num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))

# ---------------- configuration ----------------
WAVETYPES = ['kelvin', 'wmrg', 'r1', 'r2']
RAW_WAVE_FILE = '/home/users/lucalino/data/era5_wave_data/coeff_data/qvr_coefficient_{year}_k3-40_p2-10_28plev_era5.nc'   # coefficient file per year (adapt)
DN_FILE = '/home/users/lucalino/data/era5_wave_data/coeff_data/Dn.nc'
OUT_DIR = '/home/users/lucalino/data/era5_wave_data/vimd_data/'                   # data directory
LEVELS = [1000., 975., 950., 925., 900., 875., 850., 825., 800., 775., 750.,
          700., 650., 600., 550., 500., 450., 400., 350., 300., 250., 225., 200.]
P_TOP = 200.                                                     # hPa, top of the integration
LAT_MAX = 21.                                                    # computation domain (±21°); analysis keeps ±20°
TIME_CHUNK = 10                                                  # matches the dera zarr chunks


def load_wave_uv(year):
    """
    Daily-mean wave-filtered u, v for all wave types, computed from the 6-hourly projection
    coefficients. Returns {wavetype: Dataset with u, v (time, level, latitude, longitude)}.
    The reconstruction is linear and Dn does not depend on time, so taking the daily mean of
    the coefficients first gives the same result as averaging the 6-hourly u, v.
    """
    coef = xr.open_dataset(RAW_WAVE_FILE.format(year=year), chunks={'t': 4 * TIME_CHUNK})
    coef = coef[['qn', 'vn', 'rn']].sel(p=LEVELS).resample(t='1D').mean() # daily-mean ok because u,v linear in coefficients

    # the coefficients have no latitude dimension: Dn(mode, lat) supplies the meridional structure
    dn = xr.open_dataset(DN_FILE)['Dn']
    # check if coef and dn have same latitude values and restrict to 20-20 range
    dn = dn.sel(lat=abs(dn.lat) <= LAT_MAX)                           # crop the wave grid to ±LAT_MAX

    dwaves = {}
    for wavetype in WAVETYPES:
        u, v = wave_uv_from_coefficients(coef, dn, wavetype)
        ds = xr.Dataset({'u': u, 'v': v}).rename({'t': 'time', 'p': 'level', 'lat': 'latitude', 'lon': 'longitude'})
        dwaves[wavetype] = ds.assign_coords(time=ds.time.dt.floor('D'), level=ds.level.astype(float))
    return dwaves


def compute_year(dera, dwaves, year):
    """dera: ERA5 (0.25°) with q, sp, mvimd; dwaves: {wavetype: Dataset with u, v}. Returns daily Dataset."""
    target = next(iter(dwaves.values()))    # all wave datasets share one grid

    # ERA5 on 0.25°, slightly wider than the target so the 1° edge cells are fully covered
    era = dera[['q', 'u', 'v', 'sp', 'mvimd']].sel(time=str(year), level=LEVELS,
                                         latitude=slice(LAT_MAX + 1, -LAT_MAX - 1))
    era = era.assign_coords(time=era.time.dt.floor('D'), level=era.level.astype(float))
    era = era.chunk({'time': TIME_CHUNK, 'latitude': -1, 'longitude': -1, 'level': -1})

    # conservatively regrid ERA5 onto 1x1deg grid
    era1x1 = xr.Dataset({name: regrid_conservative(era[name], target) for name in ['q', 'u', 'v', 'sp', 'mvimd']})
    # add full ERA5 winds as reference ("pseudo-wavetype"), this way, the for-loop below computes 
    # vertically integrated moisture divergence for all wave types but also full ERA5 winds
    winds = {**dwaves, 'era5_full': era1x1[['u', 'v']]}    

    out = {'mvimd': era1x1.mvimd} # ERA5 horizontal moisture divergence on 1x1deg grid

    # at every wave type and at each grid point, compute the horizontal moisture flux divergence
    # associated with the wave and the advection term of this divergence (the component due to moisture 
    # gradient rather than wind gradient; wind gradient can later be computed as (vimfd - vimadv)). 
    vimfd, vimadv = [], []
    for wavetype, dw in winds.items():
        dw = dw.chunk({'time': TIME_CHUNK, 'latitude': -1, 'longitude': -1, 'level': -1})
        q, u, v = xr.align(era1x1.q, dw.u, dw.v, join='exact')   # same days, levels and grid
        # compute moisture flux divergence (mfd) and advection (adv) at each level
        terms = calc_mfd_and_adv(q, u, v)
        # surface-aware vertical integration of divergence and advection terms
        vimfd.append(column_integral(terms.mfd, era1x1.sp, P_TOP))
        vimadv.append(column_integral(terms.adv, era1x1.sp, P_TOP))
    wave = xr.DataArray(list(winds), dims='wave', name='wave')
    out['vimfd'] = xr.concat(vimfd, dim=wave)
    out['vimadv'] = xr.concat(vimadv, dim=wave)

    ds = xr.Dataset(out).transpose('wave', 'time', 'latitude', 'longitude')
    for name, long_name in [('vimfd', 'vertically integrated moisture flux divergence by wave (or full ERA5) winds'),
                            ('vimadv', 'vertically integrated moisture advection by wave (or full ERA5) winds'),
                            ('mvimd', 'ERA5 mean vertically integrated moisture divergence (regridded)')]:
           ds[name].attrs = {'long_name': long_name, 'units': 'kg m**-2 s**-1'}
    ds.attrs = {'p_top_hPa': P_TOP, 'note': 'divergence term q*div(v_wave) = vimfd - vimadv'}
    return ds


if __name__ == '__main__':
    year = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ['SLURM_ARRAY_TASK_ID'])
    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = f'{OUT_DIR}vimd_wave_daily_{year}.nc'

    dwaves = load_wave_uv(year) # dict {wavetype: Dataset with u, v (time, level, latitude, longitude)}

    # compute vertically integrated horizontal moisture flux divergence in different ways for given year
    ds = compute_year(load_era5(), dwaves, year)

    encoding = {v: {'dtype': 'float32', 'zlib': True, 'complevel': 1} for v in ds.data_vars}
    with ProgressBar():
        ds.to_netcdf(out_file, encoding=encoding)
    print(f'{out_file} saved.')
