"""
This module provides the class `ExpertKnowledgeExperimentSeriesGenerator`.
"""

# pylint: disable=relative-beyond-top-level, import-error, too-many-locals, too-few-public-methods, useless-super-delegation

from itertools import product
from math import floor
from typing import List

from data_provider.synthetic_data_generation.config.modules.experiment_series_generator_config \
    import ExpertKnowledgeExperimentSeriesGeneratorConfig
from data_provider.synthetic_data_generation.types.experiments import GeneratedExperimentSeries
from data_provider.synthetic_data_generation.types.generator_arguments \
    import ExperimentSeriesGeneratorArguments
from data_provider.synthetic_data_generation.modules.experiment_series_generators. \
    abstract_experiment_series_generator import ExperimentSeriesGenerator


class ExpertKnowledgeExperimentSeriesGenerator(ExperimentSeriesGenerator):
    """
    Class that provides functionality to generate experiment series.

    Each experiment series optimizes a set of qualities by using the expert knowledge.
    It starts with a random initialized experiment and in each step all parameters that are
    known to influence the qualities are optimized until all parameters reach a certain threshold.

    The used expert knowledge not only consists of real knowledge but also noised knowledge that
    doesn't represent the real correlations.
    """

    _config: ExpertKnowledgeExperimentSeriesGeneratorConfig

    def __init__(self, args: ExperimentSeriesGeneratorArguments) -> None:
        super().__init__(args)

    def generate_all_experiment_series(self) -> List[GeneratedExperimentSeries]:
        """
        Generates an experiment series for every quality with existing expert knowledge.

        Returns:
            List of generated experiment series
        """
        qualities: List[str] = self._pq_tuples.get_qualities_with_expert_knowledge()
        quality_pairs = self._generate_quality_pairs_to_optimize(qualities)

        experiment_series: List[GeneratedExperimentSeries] = []
        for optimized_qualities in quality_pairs:
            new_series = self._generate_single_experiment_series(optimized_qualities)
            experiment_series.append(new_series)

        return experiment_series

    def _generate_single_experiment_series(self, qualities: List[str]) -> GeneratedExperimentSeries:
        """
        Generates an experiment series for a given quality.

        Parameters:
            qualities (List): Names of the qualities that are getting optimized

        Returns:
            Generated experiment series
        """
        experiments = [self._generate_first_experiment(qualities)]

    	# Filter expert knowledge for the optimized qualities
        expert_knowledge = [(p, q) for p, q in self._pq_tuples.expert_knowledge if q in qualities]
        expert_knowledge_size = len(expert_knowledge)

        # Choose unnoised expert knowledge
        expert_knowledge = self._rng.choice(
            expert_knowledge,
            round(expert_knowledge_size * (1 - self._config.noise_proportion)),
            replace=False
        )
        expert_knowledge = [tuple(pq) for pq in expert_knowledge]

        # Generate all pq-tuples for the optimized qualities that aren't in the expert knowledge
        expert_knowledge_noise = [
            (p, q) for p, q
            in product(self._pq_tuples.selected_parameters, self._pq_tuples.selected_qualities)
            if (p, q) not in self._pq_tuples.expert_knowledge and q in qualities
        ]

        # Choose noised expert knowledge
        expert_knowledge_noise = self._rng.choice(
            expert_knowledge_noise,
            round(expert_knowledge_size * self._config.noise_proportion),
            replace=False
        )
        expert_knowledge_noise = [tuple(pq) for pq in expert_knowledge_noise]

        # Generate the pq-functions for the noised expert knowledge
        pq_functions_noise = self._pq_function_generator.generate_pq_functions(
            expert_knowledge_noise)

        expert_knowledge_all = expert_knowledge + expert_knowledge_noise
        pq_functions_all = {**self._pq_functions.pq_functions, **pq_functions_noise.pq_functions}
        adjusted_parameters = set(p for p, _ in expert_knowledge_all)

        while True:
            old_experiment = experiments[-1]
            new_parameters = old_experiment.parameters.copy()

            for parameter in adjusted_parameters:
                old_p = old_experiment.parameters[parameter]
                affected_qualities = set(q for (p, q) in expert_knowledge_all if p == parameter) & \
                                     set(qualities)

                # Calculate delta_p as mean over all qualities
                delta_p = 0
                for quality in affected_qualities:
                    if self._config.score_threshold >= \
                       self._calculate_quality_score(old_experiment.parameters, [quality]):
                        # Parameter is good enough
                        continue

                    pq_function = pq_functions_all[(parameter, quality)]
                    old_q = old_experiment.qualities[quality]
                    delta_p += pq_function.inverse_derivation(
                        old_q, last_parameter=old_p)

                # Choose next parameter value TODO: Maybe add different approaches
                delta_p /= len(affected_qualities)
                new_p = old_p - delta_p

                # Limit parameter to it's definition range
                new_p = self._sdg_config.get_parameter_by_name(parameter) \
                                        .limit_parameter_value(new_p)

                new_parameters[parameter] = new_p

            new_experiment = self._generate_experiment(
                new_parameters, old_experiment.qualities, qualities, old_experiment)
            experiments.append(new_experiment)

            if self._calculate_quality_score(new_experiment.parameters, qualities) > \
               self._calculate_quality_score(old_experiment.parameters, qualities):
                # Quit if score gets worse
                break

            if len(experiments) >= 3 and \
               self._calculate_quality_score(experiments[-1].parameters, qualities) == \
               self._calculate_quality_score(experiments[-2].parameters, qualities) == \
               self._calculate_quality_score(experiments[-3].parameters, qualities):
                # Quit if the score is the same three times
                break

            if self._config.score_threshold >= \
               self._calculate_quality_score(new_parameters, qualities):
                # Quit if score threshold is reached
                break

        if len(experiments) > self._config.max_series_size:
            # Remove experiments to match max size
            experiments_to_remove = len(experiments) - self._config.max_series_size
            if experiments_to_remove != 1:
                interval = floor((len(experiments) - 2) / (experiments_to_remove - 1))
            else:
                interval = 0
            indexes_to_remove = [i * interval for i in range(0, experiments_to_remove)]
            experiments = [e for i, e in enumerate(experiments) if i not in indexes_to_remove]

        return GeneratedExperimentSeries(
            experiments=experiments,
            generation_approach=self._config.type
        )
