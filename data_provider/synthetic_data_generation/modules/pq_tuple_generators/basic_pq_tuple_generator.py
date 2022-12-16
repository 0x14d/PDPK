"""
This module provides the class `BasicPQTupleGeneratorConfig`.
"""

# pylint: disable=relative-beyond-top-level, import-error, too-few-public-methods, too-many-locals, too-many-branches, too-many-statements, missing-function-docstring

from math import ceil, floor
from typing import List
from numpy.random import default_rng as rng
from numpy.random import Generator

from data_provider.synthetic_data_generation.config.modules.pq_tuple_generator_config \
    import BasicPQTupleGeneratorConfig, USE_ALL_PARAMETERS, USE_ALL_QUALITIES
from data_provider.synthetic_data_generation.config.basic_configs.parameter_config \
    import DefaultParameterConfig
from data_provider.synthetic_data_generation.config.basic_configs.quality_config \
    import DefaultQualityConfig
from data_provider.synthetic_data_generation.types.pq_tuple import PQTuple
from data_provider.synthetic_data_generation.types.generator_arguments \
    import PQTupleGeneratorArguments
from data_provider.synthetic_data_generation.types.pq_tuple import GeneratedPQTuples
from .abstract_pq_tuple_generator import PQTupleGenerator

class BasicPQTupleGenerator(PQTupleGenerator):
    """
    Class that provides functionality to generate pq-tuples.

    It randomly chooses pq-tuples that represent parameter-quality-correlations
    and pq-tuples that represent expert knwoledge.
    """

    _config: BasicPQTupleGeneratorConfig
    _parameters: List[str]
    _qualities: List[str]
    _rng: Generator

    def __init__(self, args: PQTupleGeneratorArguments) -> None:
        super().__init__()
        self._config = args.sdg_config.pq_tuple_generator
        self._rng = rng(self._config.seed)

        parameters: DefaultParameterConfig = args.sdg_config.parameters
        self._parameters = [p.name for p in parameters]
        qualities: DefaultQualityConfig = args.sdg_config.qualities
        self._qualities = [q.name for q in qualities]

        # Adjust configuration
        if self._config.num_parameters > len(self._parameters) or \
           self._config.num_parameters == USE_ALL_PARAMETERS:
            self._config.num_parameters = len(self._parameters)

        if self._config.num_qualities > len(self._qualities) or \
           self._config.num_qualities == USE_ALL_QUALITIES:
            self._config.num_qualities = len(self._qualities)

    def generate_pq_tuples(self) -> GeneratedPQTuples:
        # Select what parameters and qualities to use
        selected_parameters = self._rng.choice(
            a=self._parameters,
            size=self._config.num_parameters,
            replace=False,
            shuffle=False
        )
        selected_qualities = self._rng.choice(
            a=self._qualities,
            size=self._config.num_qualities,
            replace=False,
            shuffle=False
        )

        # Choose correlating pq-tuples
        pq_tuples_correlation = self._generate_correlating_pq_tuples(
            selected_parameters, selected_qualities
        )

        # Choose expert knowledge pq-tuples
        num_expert_knowledge_tuples = int(
            len(pq_tuples_correlation) * self._config.expert_knowledge_share
        )
        pq_tuples_expert_knowledge = self._rng.choice(
            a=pq_tuples_correlation,
            size=num_expert_knowledge_tuples,
            replace=False,
            shuffle=False
        )

        return GeneratedPQTuples(
            selected_parameters=selected_parameters.tolist(),
            selected_qualities=selected_qualities.tolist(),
            correlating_pq_tuples=pq_tuples_correlation,
            expert_knowledge=pq_tuples_expert_knowledge.tolist()
        )

    def _generate_correlating_pq_tuples(
        self,
        selected_parameters: List[str],
        selected_qualities: List[str]
    ) -> List[PQTuple]:
        num_correlating_tuples = int(
            len(selected_parameters) * len(selected_qualities) * self._config.pq_correlation_share
        )

        if num_correlating_tuples < len(selected_qualities):
            raise ValueError(
                'Correlation share must be higher to guarantee at least one parameter per quality!'
        )

        min_qualities = self._config.min_qualities_per_parameter
        max_qualities = self._config.max_qualities_per_parameter

        if max_qualities is None:
            max_qualities = len(selected_qualities)

        min_used_parameters = ceil(num_correlating_tuples / max_qualities)
        max_used_parameters = min(
            floor(num_correlating_tuples / min_qualities),
            len(selected_parameters)
        )

        if not min_used_parameters * min_qualities <= \
               num_correlating_tuples <= \
               min_used_parameters * max_qualities:
            raise ValueError(
                f'{num_correlating_tuples} correlating tuples can`t be achieved ' +
                f'with min / max qualities per parameter of {min_qualities} / {max_qualities}'
            )

        used_parameters = self._rng.integers(
            low=min_used_parameters,
            high=max_used_parameters,
            endpoint=True
        )
        parameter_combination = [min_qualities for _ in range(used_parameters)]
        while sum(parameter_combination) < num_correlating_tuples:
            indexes = [i for i, q in enumerate(parameter_combination) if q < max_qualities]
            index = self._rng.choice(indexes)
            parameter_combination[index] += 1
        parameters = self._rng.choice(selected_parameters, used_parameters, False)

        pq_tuples = []
        missing_qualities = selected_qualities.copy()
        for parameter, num_qualities in zip(parameters, parameter_combination):
            if len(missing_qualities) >= num_qualities:
                qualities = self._rng.choice(missing_qualities, num_qualities, False)
            else:
                qualities = self._rng.choice(
                    [q for q in selected_qualities if q not in missing_qualities],
                    num_qualities - len(missing_qualities),
                    False
                ).tolist()
                qualities += missing_qualities
            missing_qualities = [q for q in missing_qualities if q not in qualities]

            pq_tuples.extend([(parameter, q) for q in qualities])

        return pq_tuples
