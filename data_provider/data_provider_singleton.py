from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Tuple

from data_provider.abstract_data_provider import AbstractDataProvider
from data_provider.aipe_data_provider import AipeDataProvider
from data_provider.synthetic_data_provider import SyntheticDataProvider

_local_db = None


def _get_local_aipe_dp(ignore_singleton: bool=False, **kwargs):
    #use `get_data_provider` instead to get a globally unique DP object. use this with caution!"
    global _local_db
    if _local_db is None:
        _local_db = AipeDataProvider(url=AipeDataProvider.local_url,
                                     password=AipeDataProvider.local_pw,
                                     username=AipeDataProvider.local_usr)
    return _local_db


_remote_db = None


def _get_remote_aipe_dp(ignore_singleton: bool=False, **kwargs):
    global _remote_db
    if _remote_db is None:
        _remote_db = AipeDataProvider(url=AipeDataProvider.remote_url,
                                      password=AipeDataProvider.remote_pw,
                                      username=AipeDataProvider.remote_usr)
    return _remote_db

_synthetic_db = None


def _get_synthetic_aipe_dp(ignore_singleton: bool, **kwargs):
    """
    Returns the unique synthetic dataprovider.
    If `ignore_singleton` is set to True a new `SyntheticDataProvider``obejct will be returned.

    Kwargs:
        - config: Config of the synthetic data generator
    """
    global _synthetic_db
    if _synthetic_db is None:
        _synthetic_db = SyntheticDataProvider(**kwargs)
    elif ignore_singleton:
        return SyntheticDataProvider(**kwargs)
    elif kwargs is not None:
        logging.warning("kwargs will only be considered upon first initialization. reusing existing SDP!")
    return _synthetic_db


_db: Tuple[AbstractDataProvider, str] | None = None

def _get_data_provider(kind: Optional[str] = None, ignore_singleton: bool=False, **kwargs) -> Tuple[AbstractDataProvider, str]:
        # TODO explicit enumeration of ABC implementations is bad practice?!
        if kind is None:
            raise ValueError("no DP instantiated. please provide a DP type to instantiate")
        return eval(f'_get_{kind}_aipe_dp(ignore_singleton, **kwargs)'), kind

def get_data_provider(kind: Optional[str] = None, ignore_singleton: bool = False, **kwargs) -> AbstractDataProvider:
    """
    Returns the specified data provider.

    Only one dataprovider is present at a time (singleton-pattern).
    To ignore the singleton-pattern and get a data provider
    even if it's not of the current kind set the parameter `ignore_singleton` to True.
    """
    global _db

    if ignore_singleton:
        return _get_data_provider(kind, ignore_singleton, **kwargs)[0]
    if _db is None:
        _db = _get_data_provider(kind, **kwargs)
    else:
        if kind is not None and _db[1] != kind:
            raise ValueError(f"global DP already exists but is of type {_db[1]} != requested {kind}!")
    return _db[0]


_label_encoder = None


def get_label_encoder(dp: AbstractDataProvider | str | None = None, **kwargs):
    """
    get label encoder from data provider.
    label encoder is created using `AipeDataProvider.get_experiments_with_graphs()`
    kwargs: see dataprovider method
    """ ""
    global _label_encoder
    if _label_encoder is None:
        logging.info(
            f"Label encoder not created yet. calling {dp}.get_experiments_with_graph"
        )
        influential_only = kwargs.pop('influential_influences_only', True)
        if isinstance(dp, str) or dp is None:
            dp = get_data_provider(dp, **kwargs)
        dp: AbstractDataProvider
        _, _, _, _, _, _, label_encoder = dp.get_experiments_with_graphs(
            influential_influences_only=influential_only, **kwargs)
        _label_encoder = label_encoder
        if os.path.basename(os.getcwd()) == 'data_provider':
            os.chdir("..")
        with open('./obj/label_encoder.pkl', 'wb') as pkl:
            pickle.dump(_label_encoder, pkl)

    return _label_encoder