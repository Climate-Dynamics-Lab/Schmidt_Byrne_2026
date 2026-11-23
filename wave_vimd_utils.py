import sys
sys.path.append('/home/users/lucalino/')
import numpy as np
import xarray as xr
import xarray_regrid  # noqa: F401  (registers the .regrid accessor)
from utils import physical_constants

R_E = physical_constants()['r_e']
G = physical_constants()['g0']
DEG2RAD = np.pi / 180

def is_global_lon(da, lon='longitude'):
    """True if the longitudes cover the full circle (periodic in longitude)."""
    dlon = float(da[lon][1] - da[lon][0])
    return np.isclose(da.sizes[lon] * abs(dlon), 360.0)

def match_lon_convention(da, target, lon='longitude'):
    """Convert da's longitudes to the target's convention (0..360 or -180..180)."""
    if float(target[lon].min()) < 0 <= float(da[lon].min()):
        da = da.assign_coords({lon: ((da[lon] + 180) % 360) - 180})
    elif float(target[lon].min()) >= 0 > float(da[lon].min()):
        da = da.assign_coords({lon: da[lon] % 360})
    return da.sortby(lon)

# This function makes sure the regridding works well at the 0/360 seam but is probably 
# not needed because the xarray-regrid function takes care of that, too. It does not harm though,
# and this can be checked by setting n to 1 or 0 and seeing if the result changes.
def pad_longitude(da, n=3, lon='longitude'):
    """Add n wrapped-around longitudes on each side so regridding works across the seam."""
    da = da.sortby(lon)
    left = da.isel({lon: slice(-n, None)})
    left = left.assign_coords({lon: left[lon] - 360})
    right = da.isel({lon: slice(0, n)})
    right = right.assign_coords({lon: right[lon] + 360})
    return xr.concat([left, da, right], dim=lon)

def regrid_conservative(da, target, lat='latitude', lon='longitude'):
    """
    Conservatively regrid da onto the lat/lon coordinates of target.
    Handles latitude order, longitude convention and the periodic seam.
    Output has exactly the target's coordinate values and order.
    """
    da = match_lon_convention(da, target, lon).sortby(lat)
    if is_global_lon(da, lon):
        da = pad_longitude(da, lon=lon)
    tgt = xr.Dataset(coords={lat: np.sort(target[lat].values), lon: np.sort(target[lon].values)})
    out = da.regrid.conservative(tgt, latitude_coord=lat)
    return out.reindex({lat: target[lat].values, lon: target[lon].values})

def d_dlon(x, lon='longitude'):
    """∂x/∂λ per radian; centred differences, periodic across the seam if global."""
    if is_global_lon(x, lon):
        dlon = float(x[lon][1] - x[lon][0]) * DEG2RAD
        return (x.roll({lon: -1}, roll_coords=False) - x.roll({lon: 1}, roll_coords=False)) / (2 * dlon)
    return x.differentiate(lon) / DEG2RAD

def d_dlat(x, lat='latitude'):
    """∂x/∂φ per radian (works for ascending or descending latitudes)."""
    return x.differentiate(lat) / DEG2RAD

def spherical_divergence(fx, fy, lat='latitude', lon='longitude'):
    """1/(r cosφ) [∂fx/∂λ + ∂(fy cosφ)/∂φ]"""
    coslat = np.cos(np.deg2rad(fx[lat]))
    return (d_dlon(fx, lon) + d_dlat(fy * coslat, lat)) / (R_E * coslat)

def spherical_advection(q, u, v, lat='latitude', lon='longitude'):
    """v · ∇q = u/(r cosφ) ∂q/∂λ + v/r ∂q/∂φ"""
    coslat = np.cos(np.deg2rad(q[lat]))
    return u * d_dlon(q, lon) / (R_E * coslat) + v * d_dlat(q, lat) / R_E

def calc_mfd_and_adv(q, u, v):
    """
    Per-level moisture flux divergence ∇·(q v) and advection v·∇q [kg kg-1 s-1].
    The divergence term q ∇·v follows as the residual mfd - adv.
    """
    return xr.Dataset({
        'mfd': spherical_divergence(q * u, q * v),
        'adv': spherical_advection(q, u, v),
    })

def layer_thickness(level_hpa, sp, p_top_hpa, level='level'):
    """
    Pressure thickness [Pa] of the layer each level represents, between p_top and the surface.
    Layer bounds are the midpoints between neighbouring levels; the top layer starts at p_top,
    the bottom layer is bounded only by sp. Levels entirely below ground get thickness 0.
    """
    p = np.sort(level_hpa.values) * 100.0                      # top -> bottom [Pa]
    mid = 0.5 * (p[1:] + p[:-1])
    upper = np.concatenate([[p_top_hpa * 100.0], mid])
    lower = np.concatenate([mid, [np.inf]])
    coords = {level: np.sort(level_hpa.values)}
    upper = xr.DataArray(upper, dims=level, coords=coords)
    lower = xr.DataArray(lower, dims=level, coords=coords)
    return (np.minimum(lower, sp) - upper).clip(min=0)

def column_integral(x, sp, p_top_hpa, level='level'):
    """(1/g) ∫_{p_top}^{sp} x dp  ([x] kg kg-1 s-1 -> kg m-2 s-1). sp in Pa, levels in hPa."""
    dp = layer_thickness(x[level], sp, p_top_hpa, level)
    return (x * dp).sum(level, min_count=1) / G


def extract_events(field, event_times):
    """
    field:       (time, latitude, longitude)
    event_times: datetimes (lead_time, year, latitude, longitude) on the same lat/lon values
    Returns field at each grid point's own dates; NaN where a date is missing in field.time.
    """
    valid = event_times.isin(field.time.values)
    out = field.sel(time=event_times.where(valid, field.time[0])).where(valid)
    return out.drop_vars('time')


# ---------------------------------------------------------------------------
# Wave-filtered u, v from projection coefficients (qn, vn, rn) and parabolic cylinder functions Dn
# ---------------------------------------------------------------------------
# With q = Φ/c + u and r = Φ/c - u expanded as q = Σ qn Dn, r = Σ rn Dn, v = Σ vn Dn:
#   u = (q - r) / 2,  v = v
#   Kelvin:       u = q0 D0 / 2                          v = 0
#   WMRG (n=0):   u = q1 D1 / 2                          v = v0 D0
#   Rossby n:     u = (q_{n+1} D_{n+1} - r_{n-1} D_{n-1}) / 2,   v = vn Dn
WAVE_MODES = {'kelvin': -1, 'wmrg': 0, 'r1': 1, 'r2': 2}


def wave_uv_from_coefficients(coef, dn, wavetype): 
    """
    coef: Dataset with qn, vn, rn (mode, ...); dn: DataArray Dn (mode, lat).
    Returns wave-filtered (u, v). Linear in the coefficients, so it can be applied to daily means.
    """
    def m(da, n):   # select one mode and drop the scalar 'mode' coordinate so u and v can be combined
        return da.isel(mode=n).drop_vars('mode', errors='ignore')

    q, v, r, D = coef['qn'], coef['vn'], coef['rn'], dn
    n = WAVE_MODES[wavetype]
    if n == -1:
        u = m(q, 0) * m(D, 0) / 2
        return u, xr.zeros_like(u)
    if n == 0:
        return m(q, 1) * m(D, 1) / 2, m(v, 0) * m(D, 0)
    u = (m(q, n + 1) * m(D, n + 1) - m(r, n - 1) * m(D, n - 1)) / 2
    return u, m(v, n) * m(D, n)
