"""
This module provides everything that is needed for the configuration of the pq-function-generators.

To add a new generator follow these steps:
1. Create a new class that inherits from `PQFunctionGenerator` and implements the required methods
(in a new file in the `data_provider/synthetic_data_generation/modules/pq_function_generators`
directory)
2. Add a type name for the new generator to the `PQFunctionGeneratorType` enum
3. Create a new config class for the new generator (in this module). It should inherit from
`AbstractPQFunctionGeneratorConfig` and have an attribute `type` of datatype
`Literal[PQFunctionGeneratorType.<New enum member (see step 2)>]`.
4. Add a new if case to the `get_configuration` method of `PQFunctionGeneratorType` that returns
the new generator configuration if `self == PQFunctionGeneratorType.<new enum member>`
5. Add the new config class to the `PQFunctionGeneratorConfig` type by adding the new class to the
union (`Union[PQFunctionGeneratorType, ..., <new config class>]`)
"""

# pylint: disable=no-name-in-module, import-error, import-outside-toplevel, no-self-argument, too-few-public-methods, no-self-use, relative-beyond-top-level

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field, validator
from typing_extensions import Annotated

from .pq_function_config import PQFunctionConfig, parse_pq_function_config


class PQFunctionGeneratorType(str, Enum):
    """
    Enum that contains the key of every available pq-function-generator.
    It's used in the configuration to specify what generator is used.

    When adding a new generator a new enum member must be added!
    """
    SINGLE_COMPLEXITY = 'single_complexity'
    """Type of the `SingleComplexityPQFunctionGenerator`"""

    def get_configuration(self) -> AbstractPQFunctionGeneratorConfig:
        """
        Creates the matching configuration of the pq-function-generator-type.

        Returns:
            pq-function-generator configuaration with its default values
        """
        if self == PQFunctionGeneratorType.SINGLE_COMPLEXITY:
            return SingleComplexityPQFunctionGeneratorConfig()
        raise NotImplementedError(f'Missing implementation for type {self}!')


class AbstractPQFunctionGeneratorConfig(BaseModel, ABC):
    """
    Abstract class for the pq-function-generator configurations.

    When adding a new generator the associated configuration class must inherit from this class!
    """

    @abstractmethod
    def get_generator_class(self) -> Any:
        """
        Returns the class of the pq-function-generator this config is about.

        IMPORTANT: Import the returned class in this method and not outside of it!
        """

class SingleComplexityPQFunctionGeneratorConfig(AbstractPQFunctionGeneratorConfig):
    """
    Configuration of the `SingleComplexityPQFunctionGenerator`
    """
    type: Literal[PQFunctionGeneratorType.SINGLE_COMPLEXITY] \
        = PQFunctionGeneratorType.SINGLE_COMPLEXITY

    pq_function: Optional[PQFunctionConfig]
    """..."""

    seed: int = 42
    """Seed of the random number generator"""

    @validator('pq_function', always=True)
    def parse_pq_function(cls, pq_function):
        """Parses the pq_function to its default form."""
        return parse_pq_function_config(pq_function)

    def get_generator_class(self):
        from data_provider.synthetic_data_generation.modules.pq_function_generators. \
            single_complexity_pq_function_generator import SingleComplexityPQFunctionGenerator
        return SingleComplexityPQFunctionGenerator


PQFunctionGeneratorConfig = Annotated[
                            Union[
                                PQFunctionGeneratorType,
                                SingleComplexityPQFunctionGeneratorConfig
                            ],
                            Field(discriminator='type')
                        ]
"""Type alias that contains all available pq-function-generator configuration classes."""


DEFAULT_PQ_FUNCTION_GENERATOR_CONFIG: PQFunctionGeneratorType \
    = PQFunctionGeneratorType.SINGLE_COMPLEXITY
"""Default pq-function-generator type that is used if no configuration is provided."""


def parse_pq_function_generator_config(
    config: Optional[PQFunctionGeneratorConfig]
) -> AbstractPQFunctionGeneratorConfig:
    """
    Parses a pq-function-generator configuration to its default format.

    If the configuration is None the default configuration is used.

    Parameters:
        config (PQFunctionGeneratorConfig | None): config that should be parsed

    Returns:
        config in the default format (AbstractPQFunctionGeneratorConfig)
    """
    if config is None:
        config = DEFAULT_PQ_FUNCTION_GENERATOR_CONFIG
    if isinstance(config, PQFunctionGeneratorType):
        return config.get_configuration()
    return config
