"""
This module provides the class `SingleComplexityPQFunctionGenerator`.
"""

# pylint: disable=import-error, missing-function-docstring, too-few-public-methods, unused-argument

from typing import Dict, List

from numpy.random import default_rng as rng
from numpy.random import Generator

from data_provider.synthetic_data_generation.config.modules.pq_function_generator_config \
    import SingleComplexityPQFunctionGeneratorConfig
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_generation.modules.pq_function_generators. \
    abstract_pq_function_generator import PQFunctionGenerator
from data_provider.synthetic_data_generation.modules.pq_functions.abstract_pq_function \
    import PQFunction
from data_provider.synthetic_data_generation.types.generator_arguments2 \
    import PQFunctionGeneratorArguments
from data_provider.synthetic_data_generation.types.pq_function import GeneratedPQFunctions
from data_provider.synthetic_data_generation.types.pq_tuple import PQTuple


class SingleComplexityPQFunctionGenerator(PQFunctionGenerator):
    """
    Class that provides functionality to generate pq-functions.

    The underlying function between each correlating pq-tuple has the same complexity.
    If no coefficients are provided in the configuration each function gets random ones.
    """

    _config: SingleComplexityPQFunctionGeneratorConfig
    _rng: Generator
    _sdg_config: SdgConfig

    def __init__(self, args: PQFunctionGeneratorArguments) -> None:
        super().__init__(args)
        self._config = args.sdg_config.pq_function_generator
        self._rng = rng(self._config.seed)
        self._sdg_config = args.sdg_config

    def _generate_pq_functions(self,  pq_tuples: List[PQTuple]) -> GeneratedPQFunctions:
        pq_function_class = self._config.pq_function.get_function_class()

        pq_functions: Dict[PQTuple, PQFunction] = {}
        for pq_tuple in pq_tuples:
            pq_functions[pq_tuple] = pq_function_class(
                coeffs=self._config.pq_function.coeffs,
                parameter=self._sdg_config.get_parameter_by_name(pq_tuple[0]),
                quality=self._sdg_config.get_quality_by_name(pq_tuple[1]),
                rng=self._rng
            )

        return GeneratedPQFunctions(
            pq_functions=pq_functions
        )

    def regenerate_pq_functions(
        self,
        pq_functions: GeneratedPQFunctions,
        keep_direction: bool = False,
        keep_complexity: bool = False
    ) -> GeneratedPQFunctions:
        pq_function_class = self._config.pq_function.get_function_class()
        new_pq_functions: Dict[PQTuple, PQFunction] = {}
        for pq_tuple, old_function in pq_functions.pq_functions.items():
            while True:
                new_function = pq_function_class(
                    coeffs=self._config.pq_function.coeffs,
                    parameter=self._sdg_config.get_parameter_by_name(pq_tuple[0]),
                    quality=self._sdg_config.get_quality_by_name(pq_tuple[1]),
                    rng=self._rng
                )

                if keep_direction and old_function.direction != new_function.direction:
                    continue

                new_pq_functions[pq_tuple] = new_function
                break

        return GeneratedPQFunctions(
            pq_functions=new_pq_functions
        )
