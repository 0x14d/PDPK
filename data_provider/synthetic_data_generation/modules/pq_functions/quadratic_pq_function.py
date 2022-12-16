"""
This module provides the class `QuadraticPQFunction`
"""

# pylint: disable=relative-beyond-top-level, too-few-public-methods, invalid-name, missing-function-docstring, import-error, unused-argument

from numbers import Number
from typing import Callable, Optional, Tuple, Union
from numpy import isnan, sqrt
from numpy.random import Generator
from numpy.typing import ArrayLike
from numpy.polynomial.polynomial import Polynomial
from data_provider.synthetic_data_generation.config.basic_configs.parameter_config import Parameter

from data_provider.synthetic_data_generation.config.basic_configs.quality_config import Quality
from .abstract_pq_function import PQFunction

class QuadraticPQFunction(PQFunction):
    """
    Class representing a pq-function of shape f(x) = a + bx + cx^2

    See `PQFunction`
    """
    __function: Polynomial
    __derivation: Polynomial
    __inverse: Callable
    __inverse_derivation: Callable

    NUM_COEFFS: int = 3

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

        a, b , c = self._coeffs

        def inverse(y: Number, **kwargs) -> Number:
            det = sqrt(b**2 - 4 * (a - y) * c)

            if isnan(det):
                det = 0

            x = [(-b + (det * sign)) / (2 * c) for sign in [1, -1]]

            if 'return_all' in kwargs and kwargs['return_all']:
                return x

            x = self._parameter.filter_parameter_list(x)

            if 'last_parameter' in kwargs:
                return sorted(x, key=lambda x: abs(kwargs['last_parameter'] - x))[0]

            if len(x) == 1:
                return x[0]

            # Return x value thats on the right side
            vertex_x = -b / (2 * c)
            x_interval_length = self._parameter.max_value - self._parameter.min_value
            if vertex_x < self._parameter.min_value + x_interval_length / 2:
                return x[1]
            else:
                return x[0]

        self.__inverse = self._create_function_with_array_handling(inverse)

        def inverse_derivation(y: Number, **kwargs) -> Number:
            det = sqrt(b**2 - 4 * (a - y) * c)
            inv_deriv = 1 / det
            inv_deriv = [inv_deriv, -inv_deriv]

            x = self.inverse(y, return_all=True, ignore_overflow=True)
            indexes = [
                        i for i, _x in enumerate(x) \
                        if self._parameter.min_value <= _x <= self._parameter.max_value
                    ]
            x = [_x for i, _x in enumerate(x) if i in indexes]
            inv_deriv = [_inv_deriv for i, _inv_deriv in enumerate(inv_deriv) if i in indexes]

            if 'return_all' in kwargs and kwargs['return_all']:
                return inv_deriv

            if 'last_parameter' in kwargs:
                x_diff = [abs(kwargs['last_parameter'] - _x) for _x in x]
                return inv_deriv[x_diff.index(min(x_diff))]

            return inv_deriv[0]
        self.__inverse_derivation = self._create_function_with_array_handling(inverse_derivation)

    def _function(self, param: Union[Number, ArrayLike], **kwargs) -> Union[Number, ArrayLike]:
        return self.__function(param)

    def _derivation(self, param: Union[Number, ArrayLike], **kwargs) -> Union[Number, ArrayLike]:
        return self.__derivation(param)

    def _inverse(self, quality: Union[Number, ArrayLike], **kwargs) -> Union[Number, ArrayLike]:
        return self.__inverse(quality, **kwargs)

    def _inverse_derivation(
        self,
        quality: Union[Number, ArrayLike],
        **kwargs
    ) -> Union[Number, ArrayLike]:
        return self.__inverse_derivation(quality, **kwargs)

    def _generate_random_coeffs(self) -> Tuple:
        y_values = [self._quality.min_rating, self._quality.max_rating]
        self._rng.shuffle(y_values)

        vertex_x = self._rng.uniform(low=self._parameter.min_value, high=self._parameter.max_value)
        vertex_y = y_values[0]

        interval_length = self._parameter.max_value - self._parameter.min_value
        interval_threshold = interval_length * 0.1
        if vertex_x < self._parameter.min_value + interval_length / 2:
            p_x = self._parameter.max_value - self._rng.uniform(high=interval_threshold)
        else:
            p_x = self._parameter.min_value + self._rng.uniform(high=interval_threshold)
        p_y = y_values[1]

        c = (p_y - vertex_y) / (p_x - vertex_x)**2
        b = -2 * vertex_x * c
        a = c * vertex_x**2 + vertex_y

        return (a, b, c)
