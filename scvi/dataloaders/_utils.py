import logging
from typing import List, Optional, Type, Union

import h5py
import numpy as np
from anndata._core.sparse_dataset import SparseDataset
from pandas import DataFrame
from scipy.sparse import issparse
from torch import Tensor

from scvi.utils._exceptions import InvalidParameterError

logger = logging.getLogger(__name__)
ArrayLike = Union[np.ndarray, DataFrame, h5py.Dataset, SparseDataset, Tensor]


def slice_and_convert_to_numpy(
    data: ArrayLike,
    indices: Optional[List[int]] = None,
    dtype: Optional[Type] = None,
) -> np.ndarray:
    """Slice and convert data to a :class:`~numpy.ndarray` with the specified dtype."""
    if isinstance(data, h5py.Dataset):
        indices = np.arange(len(data)) if indices is None else indices
        _data = data[indices]
    elif isinstance(data, SparseDataset) or issparse(data):
        indices = np.arange(data.shape[0]) if indices is None else indices
        _data = data[indices].toarray()
    elif isinstance(data, np.ndarray):
        indices = np.arange(len(data)) if indices is None else indices
        _data = data[indices]
    elif isinstance(data, DataFrame):
        indices = np.arange(len(data)) if indices is None else indices
        _data = data.iloc[indices, :].to_numpy()
    else:
        raise InvalidParameterError(
            param="data", value=data.__class__.__name__, valid=ArrayLike
        )

    if dtype is not None:
        _data = _data.astype(dtype)

    return _data
