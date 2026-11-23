import numpy as np
import xarray as xr
from utils import load_era5, get_compressor
import warnings
from dask.array.core import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)

######### SETTINGS for gradually filling the initialized zarr store (see bottom of script) with actual data ########
START = 0
END = START + 4
####################################################################################################################

selected_year_range_store = slice(START,END)
vars = ['tcc', 'msnswrf', 'msnlwrf', 'mslhf', 'msshf']
tropics_lat = slice(20, -20)
derivatives_folder = "/gws/ssde/j25b/global_ex/lms47/data/era5_derivatives/"
store_name = 'era5_hot_day_met_vars_extreme_var=t2m_region=tropics.zarr'

def get_chunks_dict_metvars(dimensions):
    chunks = {
        "lead_time": 21,
        "year": 2,
        "latitude": 30,
        "longitude": 30,
    }
    
    if "level" in dimensions:
        chunks.update({"level": 23})

    return chunks

def get_chunks_metvars(dimensions):
    chunks = get_chunks_dict_metvars(dimensions)
    return tuple((chunks[d] for d in dimensions))

def get_encoding_metvars(dataset):
    return {
        var: {
            "compressor": get_compressor(),
            "chunks": get_chunks_metvars(dataset[var].dims),
        }
        for var in dataset.variables
    }

def get_global_attributes_metvars():
    glob_attrs = {
        "description": "Variables to elucidate the hot day meteorological background",
        "creator": "Luca Schmidt",
        "contact": "lms47@st-andrews.ac.uk",
        "source": "The dataset is based on ERA5 reanalysis data and contains various meteorological variables around the day of local annual maximum t2m values.",
        "acknowledgment": "Contains Copernicus Climate Change Service information 2020. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.",
        "valid_time_start": "1980-01-01",
        "valid_time_stop": "2023-12-31",
    }
    return glob_attrs

# load data
data_store_name = 'era5_hot_day_t2m_lead20_lag10_1980-2024.zarr'
d = xr.open_zarr(derivatives_folder + data_store_name, 
                     chunks={},
                     consolidated=True,
                     )
d = d.isel(year = selected_year_range_store)
d = d.sortby('lead_time')

# add era5 variables of interest
for var in vars:
    d[var] = load_era5()[[var]].sel(latitude = tropics_lat, time = d.time)[var]

region = {"year": selected_year_range_store}
dstore = d.drop_vars(['latitude', 'longitude', 'lead_time', 'time'])

# rechunk data
dstore = dstore.chunk(get_chunks_dict_metvars(dstore[var].dims for var in dstore.variables))
dstore.attrs = get_global_attributes_metvars()

# write data to store
dstore.to_zarr(derivatives_folder + store_name, region=region, mode="a", compute=True)

print(f'Years {start} to {end} finished.')

# ###### Code below executed once to initialize zarr store without writing actual data ########

# # load data
# data_store_name = 'era5_hot_day_t2m_lead20_lag10_1980-2024.zarr'
# d = xr.open_zarr(derivatives_folder + data_store_name, 
#                      chunks={},
#                      consolidated=True,
#                      )
# d = d.sel(year = slice(1980, 2023))
# d = d.sortby('lead_time')

# # add era5 variables of interest
# for var in vars:
#     d[var] = load_era5()[[var]].sel(latitude = tropics_lat, time = d.time)[var]

# # rechunk data
# dstore = d.chunk(get_chunks_dict_metvars(d[var].dims for var in d.variables))
# dstore.attrs = get_global_attributes_metvars()

# # write data to store
# dstore.to_zarr(derivatives_folder + store_name, mode="w", zarr_format=2, encoding=get_encoding_metvars(dstore), compute=False)