"""
This module provides the abstract class `ExperimentGenerator`.

To implement an own ExperimentGenerator follow the steps in
`data_provider.synthetic_data_generation.config.modules.experiment_generator_config.py`
"""

# pylint: disable=import-error

from abc import ABC, abstractmethod
from typing import List, Optional, Type
from numpy.random import default_rng as rng
from numpy.random import Generator

from data_provider.synthetic_data_generation.config.modules.experiment_generator_config \
    import AbstractExperimentGeneratorConfig, ExperimentGeneratorOversizeHandling
from data_provider.synthetic_data_generation.config.modules. \
    experiment_series_generator_config import AbstractExperimentSeriesGeneratorConfig
from data_provider.synthetic_data_generation.modules.experiment_series_generators. \
    abstract_experiment_series_generator import ExperimentSeriesGenerator
from data_provider.synthetic_data_generation.types.experiments \
    import GeneratedExperimentSeries, GeneratedDataset
from data_provider.synthetic_data_generation.types.experiments \
    import count_experiments as exp_len
from data_provider.synthetic_data_generation.types.generator_arguments \
    import ExperimentGeneratorArguments, ExperimentSeriesGeneratorArguments


class ExperimentGenerator(ABC):
    """
    Abstract class that provides functionality to generate multiple experiment series.
    """

    _config: AbstractExperimentGeneratorConfig
    _args: ExperimentGeneratorArguments
    _rng: Generator

    def __init__(self, args: ExperimentGeneratorArguments) -> None:
        self._config = args.sdg_config.dataset_generator
        self._args = args
        self._rng = rng(self._config.seed)

    def __call__(self) -> GeneratedDataset:
        """
        Generates experiements using the `generate_experiments` method.

        Returns:
            Generated experiments
        """
        return self.generate_experiments()

    @abstractmethod
    def generate_experiments(self) -> GeneratedDataset:
        """
        Generate experiments.

        Returns:
            Generated experiments
        """

    def _choose_included_experiment_series(
            self,
            experiment_series: List[GeneratedExperimentSeries],
            size: Optional[int] = None
        ) -> List[GeneratedExperimentSeries]:
        """
        Chooses which experiment series and experiments are included in the final dataset.

        Parameters:
            experiment_series(List[GeneratedExperimentSeries]): All experiment series
            size (int | None): Number of experiments included in the final dataset.
                If None the dataset size from the configuration is used.

        Returns:
            List of experiment series that are included in the final dataset.
        """
        if size is None:
            size = self._config.dataset_size

        self._rng.shuffle(experiment_series)

        new_experiment_series: List[GeneratedExperimentSeries] = []

        # Choose used experiment_series
        for series in experiment_series:
            new_experiment_series.append(series)
            if not self._config.use_all_experiment_series and \
                exp_len(new_experiment_series) >= size:
                break

        # Handle oversize
        def index_to_cut(length: int) -> int:
            if self._config.oversize_handling == ExperimentGeneratorOversizeHandling.CUT_FIRST:
                return 0
            if self._config.oversize_handling == ExperimentGeneratorOversizeHandling.CUT_LAST:
                return length - 1
            if self._config.oversize_handling == ExperimentGeneratorOversizeHandling.CUT_RANDOM:
                return self._rng.choice(list(range(length)))
            raise NotImplementedError(
                f'Oversize handling for {self._config.oversize_handling} is missing!'
                )

        while exp_len(new_experiment_series) > size and \
              self._config.oversize_handling != ExperimentGeneratorOversizeHandling.IGNORE:
            self._rng.shuffle(new_experiment_series)
            new_experiment_series.sort(key=len, reverse=True)
            series_to_cut = new_experiment_series[0]
            series_to_cut.experiments.pop(index_to_cut(len(series_to_cut)))

        return new_experiment_series

    def _create_experiment_series_from_class(
        self,
        experiment_series_generator_class: Type[ExperimentSeriesGenerator],
        experiment_series_config: AbstractExperimentSeriesGeneratorConfig,
        size: Optional[int] = None,
    ) -> List[GeneratedExperimentSeries]:
        """
        Creates experiment series from the specified experiment series generator
        until the specified number of experiments is reached.

        Parameters:
            experiment_series_generator_class (Type[ExperimentSeriesGenerator]):
                Class of the experiment series generator
            experiment_series_config (AbstractExperimentSeriesGeneratorConfig):
                Configuration of the experiment series generator
            size (int | None): Number of experiments included in the final dataset.
                If None the dataset size from the configuration is used.

        Returns:
            List of generated experiment series

        """
        if size is None:
            size = self._config.dataset_size

        experiment_series_generator = experiment_series_generator_class(
            ExperimentSeriesGeneratorArguments(
                experiment_series_config=experiment_series_config,
                sdg_config=self._args.sdg_config,
                pq_functions=self._args.pq_functions,
                pq_tuples=self._args.pq_tuples
            )
        )
        experiment_series: List[GeneratedExperimentSeries] = []

        while exp_len(experiment_series) < size:
            new_experiment_series = experiment_series_generator()
            experiment_series.extend(new_experiment_series)

        return experiment_series

    def _generate_dataset_from_experiment_series(
        self,
        experiment_series: List[GeneratedExperimentSeries]
    ) -> GeneratedDataset:
        """
        Converts a list of experiment series to a complete dataset

        Parameters:
            experiment_series (List[GeneratedExperimentSeries]): List of experiment series

        Returns:
            Dataset containing the experiment series
        """
        return GeneratedDataset(
            experiment_series=experiment_series,
            pq_functions=self._args.pq_functions,
            pq_tuples=self._args.pq_tuples,
            sdg_config=self._args.sdg_config
        )
