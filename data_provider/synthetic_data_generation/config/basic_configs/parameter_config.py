"""
This module provides everything that is needed for the configuration of the process parameters.

To add a new type of parameter configuration follow these steps:
1. Add the new configuration type to the `ParameterConfig` by adding the type to the union
(`Union[DefaultParameterConfig, ..., <new type>]`).
2. Modify the `parse_parameter_config_to_default_config` function to support the parsing of
the new type.
"""

# pylint: disable=no-name-in-module, too-few-public-methods, no-self-argument, no-self-use

from typing import List, Optional, Union
from pydantic import BaseModel, validator
from numpy.typing import ArrayLike


class ParameterProperties(BaseModel):
    """
    Class that contains all properties of a process parameter.
    """

    max_value: float = 255
    """Maximum rating value the quality can have"""

    min_value: float = 0
    """Minimum rating value the quality can have"""

    @validator('min_value')
    def min_value_must_be_less_than_max_value(cls, min_value, values):
        """
        Checks if `min_value` < `max_value`.
        """
        if 'max_value' in values and min_value >= values['max_value']:
            raise ValueError('must be < max_value')
        return min_value

    def filter_parameter_list(self, parameters: List[float]) -> List[float]:
        """
        Filters a list of floats to contain only values that are in the definition range
        of the parameter. (self.min_value <= p <= self.max_value)

        Parameters:
            paramters (List): List of parameter values

        Returns:
            List of parameter values that are in the definition range
        """
        return [p for p in parameters if self.min_value <= p <= self.max_value]

    def limit_parameter_value(self, parameter: float) -> float:
        """
        Limits a parameter value to fit in the definition range of the parameter.

        Parameters:
            parameter (float): Parameter value

        Returns:
            Parameter value in range [self.min_value, self.max_value]
        """
        return max(self.min_value, min(parameter, self.max_value))

    def limit_parameter_values(self, parameters: ArrayLike) -> ArrayLike:
        """
        Limits parameter values to fit in the definition range of the parameter.

        Parameters:
            parameters (ArrayLike): Parameter values

        Returns:
            Parameter values in range [self.min_value, self.max_value]
        """
        return [self.limit_parameter_value(parameter) for parameter in parameters]

class Parameter(ParameterProperties):
    """
    Class that contains all information about a process parameter.

    While `ParameterProperties` contains information about paramerters that can be shared
    between multiple parameters, `Parameter` contains the information for one specific parameter.
    """

    name: str
    """Name of the parameter"""

class ParametersWithSharedPropertiesConfig(ParameterProperties):
    """
    Parameter configuraton that contains a list of paramerter that can either be `Parameter` objects
    or `str` (parameter name).

    Parameters with no properties (only name) gain the shared parameter properties
    that are provided in this configuration.
    """

    parameters: List[Union[str, Parameter]]
    """List of parameters / parmeter names"""


DefaultParameterConfig = List[Parameter]
"""
Type alias for the parameter configuration type that is used after loading the configuration.
All other configuration types are getting parsed to this type upon loading.
"""

ParameterConfig = Union[
    DefaultParameterConfig,
    List[Union[Parameter, str]],
    ParametersWithSharedPropertiesConfig
]
"""
Type alias that contains all available parameter configuration type.

Supported configuration types:
- List of
    - `Parameter` or
    - `str` representing the parameters name
"""


def parse_parameter_config_to_default_config(config: ParameterConfig) -> DefaultParameterConfig:
    """
    Function that parses a `ParameterConfig` to the `DefaultParameterConfig`.

    When adding a new parameter config type this function has to be modified.

    Parameters:
        config (ParameterConfig): The parameter config that should be parsed.

    Returns:
        default_config (DefaultParameterConfig): Parsed config
    """
    def parse_parameter_list(
        parameters: List[Union[str, Parameter]],
        properties: Optional[dict] = None
    ) -> List[Parameter]:
        """
        Converts a list of `str` and `Parameter` to a list of `Parameter`.

        Parameters:
            parameters (list[str | Parameter]): List of parameters / parameters names
            properties (dict): Optional properties that are used to convert a parameter name to a
                parameter. If not defined the default properties are used.
        """
        parsed_parameters: List[Parameter] = []
        for param in parameters:
            if isinstance(param, str):
                # Case: param is parameter name
                parsed_parameters.append(
                    Parameter(name=param, **(properties if properties else {}))
                )
            else:
                # Case: param is Parameter object
                parsed_parameters.append(param)
        return parsed_parameters

    if isinstance(config, ParametersWithSharedPropertiesConfig):
        properties = {k: v for k, v in config.dict().items() if k != 'parameters'}
        return parse_parameter_list(config.parameters, properties)

    return parse_parameter_list(config)
