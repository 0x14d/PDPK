"""
This module provides the class `MultiTypeExperimentGenerator`.
"""

# pylint: disable=import-error, missing-function-docstring, relative-beyond-top-level, too-few-public-methods

from math import ceil
from typing import List

from data_provider.synthetic_data_generation.config.modules.experiment_generator_config \
    import MultiTypeExperimentGeneratorConfig
from data_provider.synthetic_data_generation.types.experiments \
    import GeneratedDataset, GeneratedExperimentSeries
from data_provider.synthetic_data_generation.types.generator_arguments \
    import ExperimentGeneratorArguments
from .abstract_experiment_generator import ExperimentGenerator


class MultiTypeExperimentGenerator(ExperimentGenerator):
    """
    Class that provides functionality to generate a dataset by using multiple types of
    experiment series generator.
    """

    _config: MultiTypeExperimentGeneratorConfig

    def __init__(self, args: ExperimentGeneratorArguments) -> None:
        super().__init__(args=args)

    def generate_experiments(self) -> GeneratedDataset:
        all_experiment_series: List[GeneratedExperimentSeries] = []
        for proportion, experiment_series in self._config.experiment_series.items():
            size = ceil(self._config.dataset_size * proportion)
            experiment_series = self._create_experiment_series_from_class(
                experiment_series_generator_class=experiment_series.get_generator_class(),
                experiment_series_config=experiment_series,
                size=size
            )
            experiment_series = self._choose_included_experiment_series(experiment_series, size)
            all_experiment_series.extend(experiment_series)
        all_experiment_series = self._choose_included_experiment_series(all_experiment_series)
        return self._generate_dataset_from_experiment_series(all_experiment_series)
