import xarray as xr
import h5py as h5
from arpes.endstations import SingleFileEndstation, add_endstation

def load_nexus_file(path: str) -> xr.DataArray:
    """Loads a MPES NeXus file and creates a DataArray from it.

    Args:
        path (str): The path of the .nxs file.

    Returns:
        xr.DataArray: The data read from the .nxs file.
    """    
    hf = h5.File(path, 'r')

    return xr.DataArray(
            hf['entry/data/Photoemission intensity'][:],
            coords={
                "BE": hf["entry/data/calculated_Energy"][:],
                "kx": hf["entry/data/calculated_kx"][:],
                "ky": hf["entry/data/calculated_ky"][:],
            },
            dims=["BE", "kx", "ky"],
        )

class NeXusEndstation(SingleFileEndstation):
    PRINCIPAL_NAME = "nxs"

    _TOLERATED_EXTENSIONS = {".nxs",}

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs) -> xr.Dataset:
        data = load_nexus_file(frame_path)
        return xr.Dataset({'spectrum': data})

add_endstation(NeXusEndstation)