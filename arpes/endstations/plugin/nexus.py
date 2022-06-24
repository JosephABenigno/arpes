import xarray as xr
import h5py as h5
from arpes.endstations import SingleFileEndstation, add_endstation


class NeXusEndstation(SingleFileEndstation):
    PRINCIPAL_NAME = "nxs"

    _TOLERATED_EXTENSIONS = {".nxs",}

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs):
        hf = h5.File(frame_path, 'r')
        return xr.DataArray(
            hf['entry/data/Photoemission intensity'][:],
            coords={
                "BE": hf["entry/data/calculated_Energy"][:],
                "kx": hf["entry/data/calculated_kx"][:],
                "ky": hf["entry/data/calculated_ky"][:],
            },
            dims=["BE", "kx", "ky"],
            name="Photoemission intensity"
        ).to_dataset()

add_endstation(NeXusEndstation)