"""
This module provides the abstract class `PQFunctionGenerator`.

To implement an own PQFunctionGenerator follow the steps in
`data_provider.synthetic_data_generation.config.modules.pq_function_generator_config.py`
"""

# pylint: disable=import-error

from abc import ABC, abstractmethod

from data_provider.synthetic_data_generation.types.pq_function import GeneratedPQFunctions

class PQFunctionGenerator(ABC):
    """
    Abstract class that provides functionality to generate functions representing the
    functional correlation beween a parameter-quality-tuple.
    """

    def __call__(self) -> GeneratedPQFunctions:
        """
        Generates pq-functions using the `generate_pq_functions` method.

        Returns:
            All information about the generated pq-functions
        """
        return self.generate_pq_functions()

    @abstractmethod
    def generate_pq_functions(self) -> GeneratedPQFunctions:
        """
        Generates pq-functions.

        Returns:
            All information about the generated pq-functions
        """
