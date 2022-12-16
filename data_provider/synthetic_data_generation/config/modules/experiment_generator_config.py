"""
This module provides everything that is needed for the configuration of the experiment-generators.

To add a new generator follow these steps:
1. Create a new class that inherits from `ExperimentGenerator` and implements the required methods
(in a new file in the `data_provider/synthetic_data_generation/modules/experiment_generators`
directory)
2. Add a type name for the new generator to the `ExperimentGeneratorType` enum
3. Create a new config class for the new generator (in this module). It should inherit from
`AbstractExperimentGeneratorConfig` and have an attribute `type` of datatype
`Literal[ExperimentGeneratorType.<New enum member (see step 2)>]`.
4. Add a new if case to the `get_configuration` method of `ExperimentGeneratorType` that returns
the new generator configuration if `self == ExperimentGeneratorType.<new enum member>`
5. Add the new config class to the `ExperimentGeneratorConfig` type by adding the new class to the
union (`Union[ExperimentGeneratorType, ..., <new config class>]`)
"""

# pylint: disable=no-name-in-module, import-error, import-outside-toplevel, too-few-public-methods, no-self-use, relative-beyond-top-level, no-self-argument

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Literal, Optional, Union
from pydantic import BaseModel, Field, validator
from typing_extensions import Annotated

from .experiment_series_generator_config \
    import ExperimentSeriesGeneratorConfig, parse_experiment_series_generator_config


class ExperimentGeneratorType(str, Enum):
    """
    Enum that contains the key of every available experiment-generator.
    It's used in the configuration to specify what generator is used.

    When adding a new generator a new enum member must be added!
    """
    SINGLE_TYPE = 'single_type'
    """Type of the `SingleTypeExperimentGenerator`"""

    MULTI_TYPE = 'multi_type'
    """Type of the `MultiTypeExperimentGenerator`"""

    def get_configuration(self) -> AbstractExperimentGeneratorConfig:
        """
        Creates the matching configuration of the experiment-generator-type.

        Returns:
            experiment-generator configuaration with its default values
        """
        if self == ExperimentGeneratorType.SINGLE_TYPE:
            return SingleTypeExperimentGeneratorConfig()
        if self == ExperimentGeneratorType.MULTI_TYPE:
            return MultiTypeExperimentGeneratorConfig()
        raise NotImplementedError(f'Missing implementation for type {self}!')

class ExperimentGeneratorOversizeHandling(str, Enum):
    """
    Enum that contains all methods to handle dataset oversize.
    """
    IGNORE = "ignore"
    """Ignore the oversize. The dataset will have a bigger size than specified"""

    CUT_FIRST = "cut_first"
    """Cut the first experiments from all experiment series until the specified size is reached."""

    CUT_LAST = "cut_last"
    """Cut the last experiments from all experiment series until the specified size is reached."""

    CUT_RANDOM = "cut_random"
    """Cut a random experiments from all experiment series until the specified size is reached."""

class AbstractExperimentGeneratorConfig(BaseModel, ABC):
    """
    Abstract class for the experiment-generator configurations.

    When adding a new generator the associated configuration class must inherit from this class!
    """

    dataset_size: int = 500
    """Number of experiments the dataset should contain"""

    oversize_handling: ExperimentGeneratorOversizeHandling = \
        ExperimentGeneratorOversizeHandling.IGNORE
    """Determines how dataset oversize is handled"""

    seed: int = 42
    """Seed of the random number generator"""

    use_all_experiment_series: bool = True
    """
    Determines if all generated experiment series are used.

    If True all series are used for the final dataset.

    If False the series are used until the dataset size limit is reached.
    """

    @abstractmethod
    def get_generator_class(self) -> Any:
        """
        Returns the class of the experiment-generator this config is about.

        IMPORTANT: Import the returned class in this method and not outside of it!
        """

class SingleTypeExperimentGeneratorConfig(AbstractExperimentGeneratorConfig):
    """
    Configuration of the `SingleTypeExperimentGenerator`
    """
    type: Literal[ExperimentGeneratorType.SINGLE_TYPE] \
        = ExperimentGeneratorType.SINGLE_TYPE

    experiment_series: Optional[ExperimentSeriesGeneratorConfig]

    @validator('experiment_series', always=True)
    def parse_experiment_series(cls, experiment_series):
        """Parses the experiment_series to its default form."""
        return parse_experiment_series_generator_config(experiment_series)

    def get_generator_class(self):
        from data_provider.synthetic_data_generation.modules.experiment_generators \
            .single_type_experiment_generator import SingleTypeExperimentGenerator
        return SingleTypeExperimentGenerator

class MultiTypeExperimentGeneratorConfig(AbstractExperimentGeneratorConfig):
    """
    Configuration of the `MultiTypeExperimentGenerator`
    """
    type: Literal[ExperimentGeneratorType.MULTI_TYPE] \
        = ExperimentGeneratorType.MULTI_TYPE

    experiment_series: Optional[Dict[float, ExperimentSeriesGeneratorConfig]]

    @validator('experiment_series', always=True)
    def parse_experiment_series(
        cls,
        experiment_series: Optional[Dict[float, ExperimentSeriesGeneratorConfig]]
    ):
        """Parses the experiment_series to its default form."""
        if experiment_series is None:
            return {
                1.0: parse_experiment_series_generator_config(None)
            }
        return {
            proportion: parse_experiment_series_generator_config(series)
            for proportion, series in experiment_series.items()
        }

    def get_generator_class(self):
        from data_provider.synthetic_data_generation.modules.experiment_generators. \
            multi_type_experiment_generator import MultiTypeExperimentGenerator
        return MultiTypeExperimentGenerator

ExperimentGeneratorConfig = Annotated[
                            Union[
                                ExperimentGeneratorType,
                                SingleTypeExperimentGeneratorConfig,
                                MultiTypeExperimentGeneratorConfig
                            ],
                            Field(discriminator='type')
                        ]
"""Type alias that contains all available experiment-generator configuration classes."""


DEFAULT_EXPERIMENT_GENERATOR_CONFIG: ExperimentGeneratorType \
    = ExperimentGeneratorType.SINGLE_TYPE
"""Default experiment-generator type that is used if no configuration is provided."""


def parse_experiment_generator_config(
    config: Optional[ExperimentGeneratorConfig]
) -> AbstractExperimentGeneratorConfig:
    """
    Parses an experiment-generator configuration to its default format.

    If the configuration is None the default configuration is used.

    Parameters:
        config (ExperimentGeneratorConfig | None): config that should be parsed

    Returns:
        config in the default format (AbstractExperimentGeneratorConfig)
    """
    if config is None:
        config = DEFAULT_EXPERIMENT_GENERATOR_CONFIG
    if isinstance(config, ExperimentGeneratorType):
        return config.get_configuration()
    return config
