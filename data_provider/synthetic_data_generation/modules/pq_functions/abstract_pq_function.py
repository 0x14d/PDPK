"""
This module provides the abstract class `PQFunction`

To implement an own PQFunction follow the steps in
`data_provider.synthetic_data_generation.config.modules.pq_function_config.py`
"""

# pylint: disable = invalid-name, import-error, unused-argument, no-self-use

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union
from numbers import Number
import numpy as np
from numpy.typing import ArrayLike
from numpy.random import Generator, default_rng
from data_provider.synthetic_data_generation.config.basic_configs.parameter_config import Parameter
from data_provider.synthetic_data_generation.config.basic_configs.quality_config import Quality


class PQFunction(ABC):
    """
    Abstract class representing a function that maps a configuration parameter p
    to the resulting quality q.

    Also provides methods to calcualte the derivation, inverse and derivation of the inverse.
    """

    @classmethod
    @property
    @abstractmethod
    def NUM_COEFFS(cls):
        """Number of the functions coefficients"""
        raise NotImplementedError('NUM_COEFFS is not defined!')

    _function: Callable[[Union[Number, ArrayLike], Dict[str, Any]], Union[Number, ArrayLike]]
    _derivation: Callable[[Union[Number, ArrayLike], Dict[str, Any]], Union[Number, ArrayLike]]
    _inverse: Callable[[Union[Number, ArrayLike], Dict[str, Any]], Union[Number, ArrayLike]]
    _inverse_derivation: Callable[
        [Union[Number, ArrayLike], Dict[str, Any]],
        Union[Number, ArrayLike]
    ]

    _coeffs: Tuple
    _parameter: Parameter
    _quality: Quality
    _rng: Generator

    def __init__(
        self,
        parameter: Parameter,
        quality: Quality,
        coeffs: Optional[Tuple[Number, Number]],
        rng: Optional[Generator]
    ) -> None:
        """
        Parameters:
            parameter (Parameter): Parameter this function is for
            quality (Quality): Quality this parameter is for
            coeffs (Tuple | None): Coeffs of the function. If None random coeffs will be generated
            rng (Generator | None): Random numbers generator used for coeff generation
        """
        if rng is None:
            rng = default_rng()

        self._quality = quality
        self._parameter = parameter
        self._rng = rng
        self._coeffs = self._check_coeffs(coeffs)

    def __call__(self, param: Union[Number, ArrayLike], **kwargs) -> Union[Number, ArrayLike]:
        """
        Calculates the value(s) of the function for a given parameter(s)

            Parameters:
                param (Number | Arraylike): Parameter(s) that should be evaluated
                kwargs: Additional arguments
            Returns:
                quality / qualities (Number | Arraylike): Value(s) of the function
        """
        param = self._check_parameter(param=param, param_name='param')
        quality =  self._function(param, **kwargs)
        if isinstance(quality, Number):
            return self._quality.limit_quality_rating(quality)
        return self._quality.limit_quality_ratings(quality)

    def derivation(self, param: Union[Number, ArrayLike], **kwargs) -> Union[Number, ArrayLike]:
        """
        Calculates the value(s) of the derivation for a given parameter(s)

            Parameters:
                param (Number | Arraylike): Parameter(s) that should be evaluated
                kwargs: Additional arguments

            Returns:
                quality / qualities (Number | Arraylike): Value(s) of the derivation
        """
        param = self._check_parameter(param=param, param_name='param')
        return self._derivation(param, **kwargs)

    def inverse(self, quality: Union[Number, ArrayLike], **kwargs) -> Union[Number, ArrayLike]:
        """
        Calculates the value(s) of the inverse for a given quality / qualities

            Parameters:
                quality (Number | Arraylike): Quality / qualities that should be evaluated
                kwargs: Additional arguments
                    last_parameter (float): Parameter from the last iteration if available
                    return_all (bool): If True all inverses get returned

            Returns:
                parameter(s) (Number | Arraylike): Value(s) of the inverse
        """
        quality = self._check_parameter(param=quality, param_name='quality')
        parameter = self._inverse(quality, **kwargs)
        if isinstance(parameter, Number):
            return self._parameter.limit_parameter_value(parameter)
        return parameter

    def inverse_derivation(
        self,
        quality: Union[Number, ArrayLike],
        **kwargs
    ) -> Union[Number, ArrayLike]:
        """
        Calculates the value(s) of the derivation of the inverse for a given quality / qualities

            Parameters:
                quality (Number | Arraylike): Quality / qualities that should be evaluated
                kwargs: Additional arguments
                    last_parameter (float): Parameter from the last iteration if available
                    return_all (bool): If True all inverses get returned

            Returns:
                parameter(s) (Number | Arraylike): Value(s) of the derivation of the inverse
        """
        quality = self._check_parameter(param=quality, param_name='quality')
        return self._inverse_derivation(quality, **kwargs)

    @abstractmethod
    def _function(self, param: Union[Number, ArrayLike], **kwargs) -> Union[Number, ArrayLike]:
        """
        Calculates the value(s) of the function for a given parameter(s)

            Parameters:
                param (Number | Arraylike): Parameter(s) that should be evaluated
                kwargs: Additional arguments
            Returns:
                quality / qualities (Number | Arraylike): Value(s) of the function
        """

    @abstractmethod
    def _derivation(self, param: Union[Number, ArrayLike], **kwargs) -> Union[Number, ArrayLike]:
        """
        Calculates the value(s) of the derivation for a given parameter(s)

            Parameters:
                param (Number | Arraylike): Parameter(s) that should be evaluated
                kwargs: Additional arguments

            Returns:
                quality / qualities (Number | Arraylike): Value(s) of the derivation
        """

    @abstractmethod
    def _inverse(self, quality: Union[Number, ArrayLike], **kwargs) -> Union[Number, ArrayLike]:
        """
        Calculates the value(s) of the inverse for a given quality / qualities

            Parameters:
                quality (Number | Arraylike): Quality / qualities that should be evaluated
                kwargs: Additional arguments
                    last_parameter (float): Parameter from the last iteration if available
                    return_all (bool): If True all inverses get returned

            Returns:
                parameter(s) (Number | Arraylike): Value(s) of the inverse
        """

    @abstractmethod
    def _inverse_derivation(
        self,
        quality: Union[Number, ArrayLike],
        **kwargs
    ) -> Union[Number, ArrayLike]:
        """
        Calculates the value(s) of the derivation of the inverse for a given quality / qualities

            Parameters:
                quality (Number | Arraylike): Quality / qualities that should be evaluated
                kwargs: Additional arguments
                    last_parameter (float): Parameter from the last iteration if available
                    return_all (bool): If True all inverses get returned

            Returns:
                parameter(s) (Number | Arraylike): Value(s) of the derivation of the inverse
        """

    @abstractmethod
    def _generate_random_coeffs(self) -> Tuple:
        """
        Generates random coefficients for this function

        Returns:
            Tuple with random coefficients
        """

    def _check_coeffs(self, coeffs: Optional[Tuple]) -> Tuple:
        if coeffs is None:
            return self._generate_random_coeffs()
        if len(coeffs) != self.NUM_COEFFS:
            raise ValueError(f"coeffs must have length {self.NUM_COEFFS} but had length {len(coeffs)}!")
        not_number_types = [type(c) for c in coeffs if not isinstance(c, Number)]
        if len(not_number_types) > 0:
            raise ValueError(f"All coeffs must be numbers but contain {not_number_types}!")
        return coeffs

    def _check_parameter(
        self,
        param: Union[Number, ArrayLike],
        param_name: str
    ) -> Union[Number, ArrayLike]:
        def limit_param(x: Number) -> Number:
            if param_name == 'param':
                return self._parameter.limit_parameter_value(x)
            else:
                return self._quality.limit_quality_rating(x)

        if isinstance(param, Number):
            return limit_param(param)
        else:
            not_number_types = [type(p) for p in param if not isinstance(p, Number)]
            if len(not_number_types) > 0:
                raise ValueError(
                    f'{param_name} must only contain numbers but contains {not_number_types}!'
                )
            return np.array([limit_param(p) for p in param])

    @staticmethod
    def _create_function_with_array_handling(
        function: Callable[[Number, Dict[str, Any]], Number]
    ) -> Callable[[Union[Number, ArrayLike], Dict[str, Any]], Union[Number, ArrayLike]]:
        """
        Creates a callable that can handle an arraylike input
        for a given callable that requires a numeric input.

        Paramter:
            function (callable): Callable that maps a number to another number

        Returns:
            Callable that uses the given function and can handle number and arraylike inputs
        """
        def better_function(x: Union[Number, ArrayLike], **kwargs) -> Union[Number, ArrayLike]:
            if isinstance(x, Number):
                return function(x, **kwargs)
            return [function(i) for i in x]
        return better_function
