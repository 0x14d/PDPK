"""
This module provides everything that is needed for the configuration of the pq-tuple-generators.

To add a new generator follow these steps:
1. Create a new class that inherits from `PQTupleGenerator` and implements the required methods
(in a new file in the `data_provider/synthetic_data_generation/modules/pq_tuple_generators`
directory)
2. Add a type name for the new generator to the `PQTupleGeneratorType` enum
3. Create a new config class for the new generator (in this module). It should inherit from
`AbstractPQTupleGeneratorConfig` and have an attribute `type` of datatype
`Literal[PQTupleGeneratorType.<New enum member (see step 2)>]`.
4. Add a new if case to the `get_configuration` method of `PQTupleGeneratorType` that returns
the new generator configuration if `self == PQTupleGeneratorType.<new enum member>`
5. Add the new config class to the `PQTupleGeneratorConfig` type by adding the new class to the
union (`Union[PQTupleGeneratorType, ..., <new config class>]`)
"""

# pylint: disable=no-name-in-module, import-error, import-outside-toplevel, no-self-argument, too-few-public-methods, no-self-use

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field, validator
from typing_extensions import Annotated

USE_ALL_PARAMETERS: int = 0
"""Constant that symbolizes that all available parameters should be used to generate pq-tuples"""

USE_ALL_QUALITIES: int = 0
"""Constant that symbolizes that all available qualities should be used to generate pq-tuples"""

class PQTupleGeneratorType(str, Enum):
    """
    Enum that contains the key of every available pq-tuple-generator. It's used in the configuration
    to specify what generator is used.

    When adding a new generator a new enum member must be added!
    """
    BASIC = 'basic'
    """Type of the `BasicPQTupleGenerator`"""

    def get_configuration(self) -> AbstractPQTupleGeneratorConfig:
        """
        Creates the matching configuration of the pq-tuple-generator-type.

        Returns:
            pq-tuple-generator configuaration with its default values
        """
        if self == PQTupleGeneratorType.BASIC:
            return BasicPQTupleGeneratorConfig()
        raise NotImplementedError(f'Missing implementation for type {self}!')


class AbstractPQTupleGeneratorConfig(BaseModel, ABC):
    """
    Abstract class for the pq-tuple-generator configurations.

    When adding a new generator the associated configuration class must inherit from this class!
    """

    @abstractmethod
    def get_generator_class(self) -> Any:
        """
        Returns the class of the pq-tuple-generator this config is about.

        IMPORTANT: Import the returned class in this method and not outside of it!
        """

class BasicPQTupleGeneratorConfig(AbstractPQTupleGeneratorConfig):
    """
    Configuration of the `BasicPQTupleGenerator`
    """
    type: Literal[PQTupleGeneratorType.BASIC] = PQTupleGeneratorType.BASIC

    expert_knowledge_share: float = 0.75
    """Proportion of the correlating pq-tuples that are known information (expert knowledge)"""

    num_parameters: int = USE_ALL_PARAMETERS
    """Number of parameters that should be used to generate pq-tuples (Default: all parameters)"""

    num_qualities: int = USE_ALL_QUALITIES
    """Number of qualities that should be used to generate pq-tuples (Default: all qualities)"""

    min_qualities_per_parameter: int = 1
    """Minimum number of qualities a parameter can influence"""

    max_qualities_per_parameter: Optional[int]
    """Maximum number of qualities a parameter can influence"""

    pq_correlation_share: float = 0.1
    """Proportion of the pq-tuples that should correlate to each other"""

    seed: int = 42
    """Seed of the random number generator"""

    @validator('num_parameters', 'num_qualities')
    def must_be_positive_or_zero(cls, value):
        """
        Checks if `num_parameters` and `num_qualities` is >= 0.
        """
        if value < 0:
            raise ValueError('must be positive or zero')
        return value

    @validator('pq_correlation_share', 'expert_knowledge_share')
    def must_be_between_0_and_1(cls, value):
        """
        Checks if `pq_correlation_share` and `expert_knowledge_share` are between 0 and 1.
        """
        if not 0 <= value <= 1:
            raise ValueError('must be between 0 and 1')
        return value

    def get_generator_class(self):
        from data_provider.synthetic_data_generation.modules.pq_tuple_generators. \
        basic_pq_tuple_generator import BasicPQTupleGenerator
        return BasicPQTupleGenerator


PQTupleGeneratorConfig = Annotated[
                            Union[PQTupleGeneratorType, BasicPQTupleGeneratorConfig],
                            Field(discriminator='type')
                        ]
"""Type alias that contains all available pq-tuple-generator configuration classes."""


DEFAULT_PQ_TUPLE_GENERATOR_CONFIG: PQTupleGeneratorType = PQTupleGeneratorType.BASIC
"""Default pq-tuple-generator type that is used if no configuration is provided."""


def parse_pq_tuple_generator_config(
    config: Optional[PQTupleGeneratorConfig]
) -> AbstractPQTupleGeneratorConfig:
    """
    Parses a pq-tuple-generator configuration to its default format.

    If the configuration is None the default configuration is used.

    Parameters:
        config (PQTupleGeneratorConfig | None): config that should be parsed

    Returns:
        config in the default format (AbstractPQTupleGeneratorConfig)
    """
    if config is None:
        config = DEFAULT_PQ_TUPLE_GENERATOR_CONFIG
    if isinstance(config, PQTupleGeneratorType):
        return config.get_configuration()
    return config
