"""
This module provides the class `SingleTypeExperimentGenerator`.
"""

# pylint: disable=import-error, missing-function-docstring, relative-beyond-top-level, too-few-public-methods

from typing import Type

from data_provider.synthetic_data_generation.config.modules.experiment_generator_config \
    import SingleTypeExperimentGeneratorConfig
from data_provider.synthetic_data_generation.modules.experiment_series_generators. \
    abstract_experiment_series_generator import ExperimentSeriesGenerator
from data_provider.synthetic_data_generation.types.experiments import GeneratedDataset
from data_provider.synthetic_data_generation.types.generator_arguments \
    import ExperimentGeneratorArguments
from .abstract_experiment_generator import ExperimentGenerator


class SingleTypeExperimentGenerator(ExperimentGenerator):
    """
    Class that provides functionality to generate a dataset by using a single type of
    experiment series generator.
    """

    _config: SingleTypeExperimentGeneratorConfig

    def __init__(self, args: ExperimentGeneratorArguments) -> None:
        super().__init__(args=args)

    def generate_experiments(self) -> GeneratedDataset:
        experiment_series_class: Type[ExperimentSeriesGenerator] = \
            self._config.experiment_series.get_generator_class()

        experiment_series = self._create_experiment_series_from_class(
            experiment_series_generator_class=experiment_series_class,
            experiment_series_config=self._config.experiment_series
        )

        experiment_series = self._choose_included_experiment_series(experiment_series)

        return self._generate_dataset_from_experiment_series(experiment_series)
