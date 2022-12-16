"""
This module provides everything that is needed for the configuration of the noise-generators.

To add a new generator follow these steps:
1. Create a new class that inherits from `NoiseGenerator` and implements the required methods
(in a new file in the `data_provider/synthetic_data_generation/modules/noise_generators`
directory)
2. Add a type name for the new generator to the `NoiseGeneratorType` enum
3. Create a new config class for the new generator (in this module). It should inherit from
`AbstractNoiseGeneratorConfig` and have an attribute `type` of datatype
`Literal[NoiseGeneratorType.<New enum member (see step 2)>]`.
4. Add a new if case to the `get_configuration` method of `NoiseGeneratorType` that returns
the new generator configuration if `self == NoiseGeneratorType.<new enum member>`
5. Add the new config class to the `NoiseGeneratorConfig` type by adding the new class to the
union (`Union[NoiseGeneratorType, ..., <new config class>]`)
"""

# pylint: disable=no-name-in-module, import-error, import-outside-toplevel, no-self-argument, too-few-public-methods, no-self-use

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field, validator
from typing_extensions import Annotated


class NoiseGeneratorType(str, Enum):
    """
    Enum that contains the key of every available noise-generator. It's used in the configuration
    to specify what generator is used.

    When adding a new generator a new enum member must be added!
    """
    GAUSSIAN = 'gaussian'
    """Type of the `GaussianNoiseGenerator`"""

    def get_configuration(self) -> AbstractNoiseGeneratorConfig:
        """
        Creates the matching configuration of the noise-generator-type.

        Returns:
            noise-generator configuaration with its default values
        """
        if self == NoiseGeneratorType.GAUSSIAN:
            return GaussianNoiseGeneratorConfig()
        raise NotImplementedError(f'Missing implementation for type {self}!')


class AbstractNoiseGeneratorConfig(BaseModel, ABC):
    """
    Abstract class for the noise-generator configurations.

    When adding a new generator the associated configuration class must inherit from this class!
    """

    @abstractmethod
    def get_generator_class(self) -> Any:
        """
        Returns the class of the noise-generator this config is about.

        IMPORTANT: Import the returned class in this method and not outside of it!
        """

class GaussianNoiseGeneratorConfig(AbstractNoiseGeneratorConfig):
    """
    Configuration of the `GaussianNoiseGenerator`
    """
    type: Literal[NoiseGeneratorType.GAUSSIAN] = NoiseGeneratorType.GAUSSIAN

    mean: float = 0
    """Mean of the gaussian (normal) distribution"""

    noise_proportion: float = 1.0
    """Proportion of the experiments that get noised"""

    seed: int = 42
    """Seed of the random number generator"""

    standard_deviation_proportion: float = 0.01
    """
    Proportion of the parameter / quality interval length that is used as
    standard deviation of the gaussian (normal) distribution
    """

    @validator('standard_deviation_proportion')
    def standard_deviation_must_ne_non_negative(cls, standard_deviation) -> float:
        """Checks if the standard_deviation is >= 0"""
        if standard_deviation < 0:
            raise ValueError('must be >= 0')
        return standard_deviation

    @validator('noise_proportion')
    def noise_proportion_must_be_between_0_and_1(cls, noise_proportion) -> float:
        """Checks if the noise_proportion is between 0 (not included) and 1 (included)"""
        if not 0 < noise_proportion <= 1:
            raise ValueError("must be between 0 (not included) and 1 (included)")
        return noise_proportion

    def get_generator_class(self):
        from data_provider.synthetic_data_generation.modules.noise_generators \
            .gaussian_noise_generator import GaussianNoiseGenerator
        return GaussianNoiseGenerator


NoiseGeneratorConfig = Annotated[
                            Union[NoiseGeneratorType, GaussianNoiseGeneratorConfig],
                            Field(discriminator='type')
                        ]
"""Type alias that contains all available pq-tuple-generator configuration classes."""


def parse_noise_generator_config(
    config: Optional[NoiseGeneratorConfig]
) -> Optional[AbstractNoiseGeneratorConfig]:
    """
    Parses a pq-tuple-generator configuration to its default format.

    Parameters:
        config (PQTupleGeneratorConfig | None): config that should be parsed

    Returns:
        config in the default format (AbstractPQTupleGeneratorConfig) or None if config is None
    """
    if isinstance(config, NoiseGeneratorType):
        return config.get_configuration()
    return config
