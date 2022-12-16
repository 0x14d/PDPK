"""
This module provides the class `LinearPQFunction`
"""

# pylint: disable=relative-beyond-top-level, too-few-public-methods, invalid-name, import-error, missing-function-docstring, unused-argument

from numbers import Number
from typing import Optional, Tuple, Union
from numpy.polynomial.polynomial import Polynomial
from numpy.random import Generator
from numpy.typing import ArrayLike

from data_provider.synthetic_data_generation.config.basic_configs.parameter_config import Parameter
from data_provider.synthetic_data_generation.config.basic_configs.quality_config import Quality
from .abstract_pq_function import PQFunction

class LinearPQFunction(PQFunction):
    """
    Class representing a pq-function of shape f(x) = a + bx

    See `PQFunction`
    """

    __function: Polynomial
    __derivation: Polynomial
    __inverse: Polynomial
    __inverse_derivation: Polynomial

    NUM_COEFFS: int = 2

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

        self.__function = Polynomial(coef=self._coeffs)
        self.__derivation = self.__function.deriv()

        # inverse function
        a = (-self._coeffs[0]) / self._coeffs[1]
        b = 1 / self._coeffs[1]
        self.__inverse = Polynomial([a, b])

        self.__inverse_derivation = self.__inverse.deriv()

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

    def _generate_random_coeffs(self) -> Tuple[Number, Number]:
        interval_threshold = (self._parameter.max_value - self._parameter.min_value) * 0.1

        x = [
            self._rng.uniform(
                low=self._parameter.min_value,
                high=self._parameter.min_value + interval_threshold
            ),
            self._rng.uniform(
                low=self._parameter.max_value - interval_threshold,
                high=self._parameter.max_value
            )
        ]
        y = [self._quality.min_rating, self._quality.max_rating]
        self._rng.shuffle(x)

        b = (y[0] - y[1]) / (x[0] - x[1])
        a = (b * x[0] - y[0]) * -1

        return (a, b)
