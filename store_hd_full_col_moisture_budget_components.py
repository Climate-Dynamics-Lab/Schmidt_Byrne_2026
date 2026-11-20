import numpy as np
import xarray as xr
from utils import load_era5, get_compressor, time_derivative
import warnings
from dask.array.core import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)

def get_chunks_dict(dimensions):
    chunks = {
        "lead_time": 21,
        "year": 44,
        "latitude": 10,
        "longitude": 10,
    }
    
    if "level" in dimensions:
        chunks.update({"level": 23})

    return chunks

def get_chunks(dimensions):
    chunks = get_chunks_dict(dimensions)
    return tuple((chunks[d] for d in dimensions))

def get_encoding(dataset):
    return {
        var: {
            "compressor": get_compressor(),
            "chunks": get_chunks(dataset[var].dims),
        }
        for var in dataset.variables
    }

def get_global_attributes():
    glob_attrs = {
        "description": "Hot day full column moisture budget components",
        "creator": "Luca Schmidt",
        "contact": "lms47@st-andrews.ac.uk",
        "source": "The dataset is based on ERA5 reanalysis data and contains full column moisture budget components for a number of days around the day of local annual maximum t2m values.",
        "acknowledgment": "Contains Copernicus Climate Change Service information 2020. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.",
        "valid_time_start": "1980-01-01",
        "valid_time_stop": "2023-12-31",
}
    return glob_attrs

derivatives_folder = "/gws/ssde/j25b/global_ex/lms47/data/era5_derivatives/"
tropics_lat = slice(20, -20)


# load data (exclude 2024 due to december hot days)
extreme_dates_data_name = 'era5_hot_day_t2m_lead20_lag10_1980-2024.zarr' 
dex = xr.open_zarr(derivatives_folder + extreme_dates_data_name, 
                     chunks={},
                     consolidated=True,
                     )
dex = dex.sel(year = slice(1980, 2023))

dextra = load_era5()[['tcwv']].sel(latitude = tropics_lat)
dextra['d_tcwv_dt'] = time_derivative(dextra['tcwv']).compute()

# expand dataset by moisture budget components
dex['tcwv'] = dextra['tcwv'].sel(time = dex.time)
dex['d_tcwv_dt'] = dextra['d_tcwv_dt'].sel(time = dex.time)
dex['q2m'] = load_era5()['q2m'].sel(latitude = tropics_lat, time = dex.time)
dex['mer'] = load_era5()['mer'].sel(latitude = tropics_lat, time = dex.time)
dex['mtpr'] = load_era5()['mtpr'].sel(latitude = tropics_lat, time = dex.time)
dex['mvimd'] = load_era5()['mvimd'].sel(latitude = tropics_lat, time = dex.time)

# rechunk data and add global attributes
dex = dex.chunk(get_chunks_dict(dex[var].dims for var in dex.variables))
dex.attrs = get_global_attributes()

# write data to store
dex.to_zarr(derivatives_folder + 'era5_hot_day_full_col_mb_components_extreme_var=t2m_region=tropics.zarr', mode="w", zarr_format=2, encoding=get_encoding(dex), compute=True)