"""
This module provides the abstract class `PQFunctionGenerator`.

To implement an own PQFunctionGenerator follow the steps in
`data_provider.synthetic_data_generation.config.modules.pq_function_generator_config.py`
"""

# pylint: disable=import-error

from abc import ABC, abstractmethod
from typing import List, Optional

from data_provider.synthetic_data_generation.types.generator_arguments2 \
    import PQFunctionGeneratorArguments
from data_provider.synthetic_data_generation.types.pq_function import GeneratedPQFunctions
from data_provider.synthetic_data_generation.types.pq_tuple import PQTuple


class PQFunctionGenerator(ABC):
    """
    Abstract class that provides functionality to generate functions representing the
    functional correlation beween a parameter-quality-tuple.
    """
    _pq_tuples: List[PQTuple]

    def __init__(self, args: PQFunctionGeneratorArguments) -> None:
        self._pq_tuples = args.pq_tuples.correlating_pq_tuples

    def __call__(self) -> GeneratedPQFunctions:
        """
        Generates pq-functions using the `generate_pq_functions` method.

        Returns:
            All information about the generated pq-functions
        """
        return self.generate_pq_functions()

    def generate_pq_functions(
        self,
        pq_tuples: Optional[List[PQTuple]] = None
    ) -> GeneratedPQFunctions:
        """
        Generates pq-functions for the specified tuples.
        If no tuples are provided the generated pq_tuples will be used.

        Returns:
            All information about the generated pq-functions
        """
        if pq_tuples is None:
            pq_tuples = self._pq_tuples
        return self._generate_pq_functions(pq_tuples)

    @abstractmethod
    def regenerate_pq_functions(
        self,
        pq_functions: GeneratedPQFunctions,
        keep_direction: bool = False,
        keep_complexity: bool = False
    ) -> GeneratedPQFunctions:
        """
        Generates the provided pq_functions again

        Parameters:
            - pq_functions (`GeneratedPQFunctions`): pq-functions that should be generated again
            - keep_direction (bool, default: False): whether the newly generated function should have the same direction as the old ones
            - keep_complexity (bool, default: False): whether the newly generated function should have the same complexity as the old ones

        Returns:
            Regernated pq-functions
        """

    @abstractmethod
    def _generate_pq_functions(self, pq_tuples: List[PQTuple]) -> GeneratedPQFunctions:
        """
        Generates pq-functions for the specified tuples.

        Returns:
            All information about the generated pq-functions
        """
