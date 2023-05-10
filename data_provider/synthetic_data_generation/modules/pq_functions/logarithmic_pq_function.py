"""
This module provides the class `LogarithmicPQFunction`
"""

# pylint: disable=relative-beyond-top-level, too-few-public-methods, invalid-name, missing-function-docstring, import-error, no-self-use, unused-argument

from sys import float_info
from math import log
from numbers import Number
from typing import Callable, Optional, Tuple, Union

from numpy.random import Generator
from numpy.typing import ArrayLike

from data_provider.synthetic_data_generation.config.basic_configs.parameter_config import Parameter
from data_provider.synthetic_data_generation.config.basic_configs.quality_config import Quality
from .abstract_pq_function import PQFunction

class LogarithmicPQFunction(PQFunction):
    """
    Class representing a pq-function of shape f(x) = alog_b(x - c) + d

    See `PQFunction`
    """

    __function: Callable
    __derivation: Callable
    __inverse: Callable
    __inverse_derivation: Callable

    NUM_COEFFS = 4

    def __init__(
        self,
        parameter: Parameter,
        quality: Quality,
        coeffs: Optional[Tuple[Number, Number]],
        rng: Optional[Generator]
    ) -> None:
        super().__init__(
            parameter=parameter,
            quality=quality,
            coeffs=coeffs,
            rng=rng
        )

        a, b, c, d = self._coeffs

        def function(x: Number) -> Number:
            if x < c:
                # if x - c <= 0 use the smallest possible value
                return a * log(float_info.min, b) + d
            return a * log(x - c, b) + d
        self.__function = self._create_function_with_array_handling(function)

        def derivation(x: Number) -> Number:
            return a / (log(b) * (x - c))
        self.__derivation = self._create_function_with_array_handling(derivation)

        def inverse(x: Number) -> Number:
            i = b ** ((x - d) / a) + c
            if i == 0:
                # if inverse is 0 a floating point error occured
                return float_info.min
            return i
        self.__inverse = self._create_function_with_array_handling(inverse)

        def inverse_derivation(x: Number) -> Number:
            return (b ** ((x - d) / a) * log(b)) / a
        self.__inverse_derivation = self._create_function_with_array_handling(inverse_derivation)

    @property
    def direction(self) -> int:
        return 1 if self._coeffs[1] >= 2 else -1

    def _function(self, param: Union[Number, ArrayLike], **kwargs) -> Union[Number, ArrayLike]:
        return self.__function(param)

    def _derivation(self, param: Union[Number, ArrayLike], **kwargs) -> Union[Number, ArrayLike]:
        return self.__derivation(param)

    def _inverse(self, quality: Union[Number, ArrayLike], **kwargs) -> Union[Number, ArrayLike]:
        return self.__inverse(quality)

    def _inverse_derivation(
        self,
        quality: Union[Number, ArrayLike],
        **kwargs
    ) -> Union[Number, ArrayLike]:
        return self.__inverse_derivation(quality)

    def _generate_random_coeffs(self) -> Tuple:
        qualities = [self._quality.min_rating, self._quality.max_rating]
        self._rng.shuffle(qualities)

        interval_threshold = (self._parameter.max_value - self._parameter.min_value) * 0.1
        p1_x = self._rng.uniform(
            low=self._parameter.min_value,
            high=self._parameter.min_value + interval_threshold
        )

        c = p1_x - 1
        d = qualities[0]

        if d == self._quality.min_rating:
            b = self._rng.uniform(low=2, high=10)
        else:
            b = self._rng.uniform(low=float_info.min, high=1)

        p2_x = self._rng.uniform(
            low=self._parameter.max_value - interval_threshold,
            high=self._parameter.max_value
        )
        a = (qualities[1] - d) / log(p2_x - c, b)

        return (a, b, c, d)
