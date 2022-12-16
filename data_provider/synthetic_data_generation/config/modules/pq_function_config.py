"""
This module provides everything that is needed for the configuration of the pq-functions.

To add a new function follow these steps:
1. Create a new class that inherits from `PQFunction` and implements the required methods
(in a new file in the `data_provider/synthetic_data_generation/modules/pq_functions`
directory)
2. Add a type name for the new function to the `PQFunctionType` enum
3. Create a new config class for the new function (in this module). It should inherit from
`AbstractPQFunctionConfig` and have an attribute `type` of datatype
`Literal[PQFunctionType.<New enum member (see step 2)>]`.
4. Add a new if case to the `get_configuration` method of `PQFunctionType` that returns
the new generator configuration if `self == PQFunctionType.<new enum member>`
5. Add the new config class to the `PQFunctionConfig` type by adding the new class to the
union (`Union[PQFunctionType, ..., <new config class>]`)
"""

# pylint: disable=no-name-in-module, import-error, import-outside-toplevel, no-self-argument, too-few-public-methods, no-self-use, missing-class-docstring

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field
from typing_extensions import Annotated


class PQFunctionType(str, Enum):
    """
    Enum that contains the key of every available pq-function. It's used in the configuration
    to specify what functions are used.

    When adding a new function a new enum member must be added!
    """
    LINEAR = 'linear'
    """Type of the `LinearPQFunction`"""

    QUADRATIC = 'quadratic'
    """Type of the `QuadraticPQFunction`"""

    LOGARITHMIC = 'logarithmic'
    """Type of the `LogarithmicPQFunction`"""

    def get_configuration(self) -> AbstractPQFunctionConfig:
        """
        Creates the matching configuration of the pq-function-type.

        Returns:
            pq-function configuaration with its default values
        """
        if self == PQFunctionType.LINEAR:
            return LinearPQFunctionConfig()
        if self == PQFunctionType.QUADRATIC:
            return QuadraticPQFunctionConfig()
        if self == PQFunctionType.LOGARITHMIC:
            return LogarithmicPQFunctionConfig()
        raise NotImplementedError(f'Missing implementation for type {self}!')


class AbstractPQFunctionConfig(BaseModel, ABC):
    """
    Abstract class for the pq-function configurations.

    When adding a new function the associated configuration class must inherit from this class!
    """

    coeffs: Optional[Tuple] = None
    """
    Coefficients of the function.

    When inheriting from this class this property should be overwritten with the exact type.
    """

    @abstractmethod
    def get_function_class(self) -> Any:
        """
        Returns the pq-function class this config is about.

        IMPORTANT: Import the returned class in this method and not outside of it!
        """

class LinearPQFunctionConfig(AbstractPQFunctionConfig):
    """
    Configuration of the `LinearPQFunction`
    """
    type: Literal[PQFunctionType.LINEAR] = PQFunctionType.LINEAR

    coeffs: Optional[Tuple[float, float]]
    """Coefficients (a, b) of the linear function f(x) = a + bx"""

    def get_function_class(self):
        from data_provider.synthetic_data_generation.modules.pq_functions.linear_pq_function \
            import LinearPQFunction
        return LinearPQFunction

class QuadraticPQFunctionConfig(AbstractPQFunctionConfig):
    """
    Configuration of the `QuadraticPQFunction`
    """
    type: Literal[PQFunctionType.QUADRATIC] = PQFunctionType.QUADRATIC

    coeffs: Optional[Tuple[float, float, float]]
    """Coefficients (a, b, c) of the linear function f(x) = a + bx + cx^2"""

    def get_function_class(self):
        from data_provider.synthetic_data_generation.modules.pq_functions.quadratic_pq_function \
            import QuadraticPQFunction
        return QuadraticPQFunction

class LogarithmicPQFunctionConfig(AbstractPQFunctionConfig):
    """
    Configuration of the `LogarithmicPQFunction`
    """
    type: Literal[PQFunctionType.LOGARITHMIC] = PQFunctionType.LOGARITHMIC

    coeffs: Optional[Tuple[float, float, float, float]]
    """Coefficients (a, b, c, d) of the logarithmic function alog_b(x - c) + d"""

    def get_function_class(self):
        from data_provider.synthetic_data_generation.modules.pq_functions.logarithmic_pq_function \
            import LogarithmicPQFunction
        return LogarithmicPQFunction

PQFunctionConfig = Annotated[
                            Union[
                                PQFunctionType,
                                LinearPQFunctionConfig,
                                QuadraticPQFunctionConfig,
                                LogarithmicPQFunctionConfig
                            ],
                            Field(discriminator='type')
                        ]
"""Type alias that contains all available pq-function configuration classes."""


DEFAULT_PQ_FUNCTION_CONFIG: PQFunctionType = PQFunctionType.LINEAR
"""Default pq-function type that is used if no configuration is provided."""


def parse_pq_function_config(
    config: Optional[PQFunctionConfig]
) -> AbstractPQFunctionConfig:
    """
    Parses a pq-function configuration to its default format.

    If the configuration is None the default configuration is used.

    Parameters:
        config (PQFunctionConfig | None): config that should be parsed

    Returns:
        config in the default format (AbstractPQFunctionConfig)
    """
    if config is None:
        config = DEFAULT_PQ_FUNCTION_CONFIG
    if isinstance(config, PQFunctionType):
        return config.get_configuration()
    return config
