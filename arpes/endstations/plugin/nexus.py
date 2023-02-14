import xarray as xr
import h5py
import numpy as np
from arpes.endstations import SingleFileEndstation, add_endstation


class NeXusEndstation(SingleFileEndstation):
    """An endstation for reading arpes data from a nexus file."""

    PRINCIPAL_NAME = "nxs"

    _TOLERATED_EXTENSIONS = {
        ".nxs",
    }

    def load_nexus_file(self, path: str) -> xr.DataArray:
        """Loads a MPES NeXus file and creates a DataArray from it.

        Args:
            path (str): The path of the .nxs file.

        Returns:
            xr.DataArray: The data read from the .nxs file.
        """
        with h5py.File(path, "r") as h5file:
            return xr.DataArray(
                h5file["entry/data/Photoemission intensity"][:],
                coords={
                    "tpp": np.squeeze(h5file["entry/data/calculated_Tpp"][:]),
                    "BE": np.squeeze(np.transpose(h5file["entry/data/calculated_Energy"][:])),
                    "kx": np.squeeze(h5file["entry/data/calculated_kx"][:]),
                    "ky": np.squeeze(h5file["entry/data/calculated_ky"][:]),
                },
                dims=["delay", "eV", "kx", "ky"],
            )

    def load_single_frame(
        self, frame_path: str = None, scan_desc: dict = None, **kwargs
    ) -> xr.Dataset:
        data = self.load_nexus_file(frame_path)
        return xr.Dataset({"spectrum": data})


add_endstation(NeXusEndstation)
