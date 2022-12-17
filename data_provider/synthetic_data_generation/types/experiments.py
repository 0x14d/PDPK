"""
This module provides type definitions regarding experiments.
"""

# pylint: disable=no-name-in-module, too-few-public-methods, import-error

from __future__ import annotations

from functools import reduce
from itertools import count
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from copy import deepcopy
import pandas as pd
import numpy as np

from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_generation.types.pq_function import GeneratedPQFunctions
from data_provider.synthetic_data_generation.types.pq_tuple import GeneratedPQTuples

def count_experiments(experiment_series: List[GeneratedExperimentSeries]) -> int:
    """
    Counts the number of experiments that are contained in a list of experiment series.
    """
    return sum(len(series) for series in experiment_series)

@dataclass
class GeneratedExperiment:
    """
    Class that contains all information about a single generated experiment.
    """
    parameters: Dict[str, float]
    """Dictionary containing the parameter values of the experiment"""

    qualities: Dict[str, float]
    """Dictionary containing the quality ratings of the experiment"""

    experiment_id: int = field(default_factory=count().__next__)
    """Unique id of the experiment"""

    def to_pandas_series(self) -> pd.Series:
        """
        Converts the experiment to a pandas Series
        """
        return pd.Series(self.to_dict(), name=self.experiment_id)

    def to_dict(self) -> Dict[str, float]:
        """
        Converts the experiment to a dict
        """
        return {**self.parameters, **self.qualities}

    def copy(self) -> GeneratedExperiment:
        """
        Creates a deep copy of the experient
        """
        return deepcopy(self)

@dataclass
class GeneratedExperimentSeries:
    """
    Class that contains all information about a single generated experiment series.
    """

    experiments: List[GeneratedExperiment]
    """List of generated experiments"""

    generation_approach: str
    """Generator used to generate this experiment series"""

    optimized_qualities: List[str] = field(default_factory=lambda: [])
    """List of qualities that get optimized in this experiment series"""

    experiment_series_id: int = field(default_factory=count().__next__)
    """Unique id of the experiment series"""

    def __len__(self):
        return len(self.experiments)

    def __iter__(self):
        return iter(self.experiments)

    def to_pandas_dataframe(self) -> pd.DataFrame:
        """
        Converts the experiment series to a pandas Dataframe
        """
        dataframe = pd.DataFrame()
        for experiment in self.experiments:
            dataframe = dataframe.append(experiment.to_pandas_series())
        return dataframe

    def to_csv(self, path: Optional[str]=None) -> Optional[str]:
        """
        Converts the experiment series to csv format

        Parameters:
            path (str or None): Path where the csv file gets saved to. If None the csv string will be returned instead

        Returns:
            Csv string if path is None, else None
        """
        return self.to_pandas_dataframe().to_csv(path)

    def copy(self) -> GeneratedExperimentSeries:
        """
        Creates a deep copy of the experient series
        """
        return deepcopy(self)

    def filter_experiments_by_id(self, ids: List[int]) -> GeneratedExperimentSeries:
        """
        Creates an experiment series only containing experiments with the specified ids.
        """
        copy = self.copy()
        copy.experiments = [e for e in copy.experiments if e.experiment_id in ids]
        return copy

@dataclass
class GeneratedDataset:
    """
    Class that contains all information about the experiments that were generated by a
    `ExperimentGenerator`.
    """

    experiment_series: List[GeneratedExperimentSeries]
    """List of generated experiment series"""

    pq_functions: GeneratedPQFunctions
    """Information about the generated pq-functions"""

    pq_tuples: GeneratedPQTuples
    """Information about the generated pq-tuples"""

    sdg_config: SdgConfig
    """Configuration used to create this dataset"""

    @property
    def all_experiments(self) -> List[GeneratedExperiment]:
        """
        A list containing all experiments of the dataset.
        """
        return reduce(
            lambda all_exps, series: all_exps + series.experiments,
            self.experiment_series,
            []
        )

    @property
    def num_experiment_series(self) -> int:
        """Number of experiment series in the dataset"""
        return len(self.experiment_series)

    @property
    def avg_experiment_series_iterations(self) -> float:
        """Average number of iterations per experiment series"""
        return np.average([len(s) for s in self.experiment_series])

    def __len__(self):
        return count_experiments(self.experiment_series)

    def __iter__(self):
        return iter(self.experiment_series)

    def to_pandas_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas Dataframe
        """
        experiment_dfs = [series.to_pandas_dataframe() for series in self.experiment_series]
        return pd.concat(experiment_dfs).sort_index()

    def to_csv(self, path: Optional[str]=None) -> Optional[str]:
        """
        Converts the dataset to csv format

        Parameters:
            path (str or None): Path where the csv file gets saved to.
            If None the csv string will be returned instead

        Returns:
            Csv string if path is None, else None
        """
        return self.to_pandas_dataframe().to_csv(path)

    def copy(self) -> GeneratedDataset:
        """
        Creates a deep copy of the dataset
        """
        return deepcopy(self)

    def get_all_experiment_series_for_quality(
        self,
        quality: str
    ) -> List[GeneratedExperimentSeries]:
        """
        Returns a list containing all experiment series for a specific quality.
        """
        return [s for s in self.experiment_series if quality in s.optimized_qualities]

    def filter_experiments_by_id(self, ids: List[int]) -> GeneratedDataset:
        """
        Creates a dataset only containing experiments with the specified ids.
        """
        copy = self.copy()
        copy.experiment_series = [s.filter_experiments_by_id(ids) for s in copy.experiment_series]
        copy.experiment_series = [s for s in copy.experiment_series if len(s) > 0]
        return copy