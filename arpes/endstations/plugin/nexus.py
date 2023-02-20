from typing import Any, Dict, Union
import xarray as xr
import h5py
import numpy as np
from arpes.endstations import SingleFileEndstation, add_endstation
from arpes.config import ureg

__all__ = ["NeXusEndstation"]

class NeXusEndstation(SingleFileEndstation):
    """An endstation for reading arpes data from a nexus file."""

    PRINCIPAL_NAME = "NXmpes"

    _TOLERATED_EXTENSIONS = {
        ".nxs",
    }


    def load_nexus_file(self, filepath: str, entry_name: str = "entry") -> xr.DataArray:
        """Loads a MPES NeXus file and creates a DataArray from it.

        Args:
            filepath (str): The path of the .nxs file.

        Returns:
            xr.DataArray: The data read from the .nxs file.
        """
        def write_value(name: str, dataset: h5py.Dataset):
            if str(dataset.dtype) == 'bool':
                attributes[name] = bool(dataset[()])
            elif dataset.dtype.kind in 'iufc':
                attributes[name] = dataset[()]
                if 'units' in dataset.attrs:
                    attributes[name] = attributes[name] * ureg(dataset.attrs['units'])
            elif dataset.dtype.kind in "O" and dataset.shape == ():
                attributes[name] = dataset[()].decode()

        def is_valid_metadata(name: str) -> bool:
            invalid_end_paths = ['depends_on']
            invalid_start_paths = ['data', 'process']
            for invalid_path in invalid_start_paths:
                if name.startswith(invalid_path):
                    return False
            for invalid_path in invalid_end_paths:
                if name.endswith(invalid_path):
                    return False
            return True

        def parse_attrs(name: str, dataset: Union[h5py.Dataset, h5py.Group]):
            short_path = name.split('/', 1)[-1]
            if isinstance(dataset, h5py.Dataset) and is_valid_metadata(short_path):
                write_value(short_path, dataset)

        data_path = f"/{entry_name}/data"
        with h5py.File(filepath, "r") as h5file:
            attributes = {}
            h5file.visititems(parse_attrs)
            return xr.DataArray(
                h5file[f"/{data_path}/data"][:],
                coords={
                    "delay": h5file[f"{data_path}/delay"][:],
                    "eV": np.transpose(h5file[f"{data_path}/energy"][:]),
                    "kx": h5file[f"{data_path}/kx"][:],
                    "ky": h5file[f"{data_path}/ky"][:],
                },
                dims=["kx", "ky", "eV", "delay"],
                attrs=attributes
            )

    def load_single_frame(
        self, frame_path: str = None, scan_desc: dict = None, **kwargs
    ) -> xr.Dataset:
        data = self.load_nexus_file(frame_path)
        return xr.Dataset({"spectrum": data}, attrs=data.attrs)


add_endstation(NeXusEndstation)
