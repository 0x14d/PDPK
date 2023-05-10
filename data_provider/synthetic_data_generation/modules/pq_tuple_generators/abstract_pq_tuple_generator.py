"""
This module provides the abstract class `PQTupleGenerator`.

To implement an own PQTupleGenerator follow the steps in
`data_provider.synthetic_data_generation.config.modules.pq_tuple_generator_config.py`
"""

# pylint: disable=import-error

from abc import ABC, abstractmethod

from numpy.random import default_rng as rng
from numpy.random import Generator

from data_provider.synthetic_data_generation.config.modules.pq_tuple_generator_config \
    import AbstractPQTupleGeneratorConfig
from data_provider.synthetic_data_generation.types.generator_arguments \
    import PQTupleGeneratorArguments
from data_provider.synthetic_data_generation.types.pq_tuple import GeneratedPQTuples


class PQTupleGenerator(ABC):
    """
    Abstract class that provides functionality to generate parameter-quality-tuples representing the
    p-q-pairs that have an influential connection.
    """

    _config: AbstractPQTupleGeneratorConfig
    _rng: Generator

    def __init__(self, args: PQTupleGeneratorArguments) -> None:
        self._config = args.sdg_config.pq_tuple_generator
        self._rng = rng(self._config.seed)

    def __call__(self) -> GeneratedPQTuples:
        """
        Generates pq-tuples using the `generate_pq_tuples` method.

        Returns:
            All information about the generated pq-tuples
        """
        return self.generate_pq_tuples()

    @abstractmethod
    def generate_pq_tuples(self) -> GeneratedPQTuples:
        """
        Generates pq-tuples.

        Returns:
            All information about the generated pq-tuples
        """
