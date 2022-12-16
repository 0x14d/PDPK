"""
This module provides the class `ExpertKnowledgeExperimentSeriesGenerator`.
"""

# pylint: disable=relative-beyond-top-level, import-error, too-many-locals, too-few-public-methods, useless-super-delegation

from math import floor
from typing import List
from data_provider.synthetic_data_generation.config.modules.experiment_series_generator_config \
    import ExpertKnowledgeExperimentSeriesGeneratorConfig
from data_provider.synthetic_data_generation.types.experiments import GeneratedExperimentSeries
from data_provider.synthetic_data_generation.types.generator_arguments \
    import ExperimentSeriesGeneratorArguments
from .abstract_experiment_series_generator import ExperimentSeriesGenerator


class ExpertKnowledgeExperimentSeriesGenerator(ExperimentSeriesGenerator):
    """
    Class that provides functionality to generate experiment series.

    Each experiment series optimizes a set of qualities by using the expert knowledge.
    It starts with a random initialized experiment and in each step all parameters that are
    known to influence the qualities are optimized until all parameters reach a certain threshold.
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

        adjusted_parameters = self._pq_tuples.get_expert_knowledge_for_qualities(qualities)
        while True:
            old_experiment = experiments[-1]
            new_parameters = old_experiment.parameters.copy()

            for parameter in adjusted_parameters:
                old_p = old_experiment.parameters[parameter]
                affected_qualities = set(self._pq_tuples.get_expert_knowledge_for_parameter(parameter)) \
                                     & set(qualities)

                # Calculate delta_p as mean over all qualities
                delta_p = 0
                for quality in affected_qualities:
                    pq_function = self._pq_functions.pq_functions[(parameter, quality)]

                    if self._config.score_threshold >= \
                       self._calculate_quality_score(old_experiment.parameters, [quality]):
                        # Parameter is good enough
                        continue

                    old_q = old_experiment.qualities[quality]
                    delta_p += pq_function.inverse_derivation(old_q, last_parameter=old_p)

                # Choose next parameter value TODO: Maybe add different approaches
                delta_p /= len(affected_qualities)
                new_p = old_p - delta_p

                # Limit parameter to it's definition range
                new_p = self._sdg_config.get_parameter_by_name(parameter) \
                                        .limit_parameter_value(new_p)

                new_parameters[parameter] = new_p

            if len(experiments) > 1 and \
               self._calculate_quality_score(new_parameters, qualities) >= \
               self._calculate_quality_score(old_experiment.parameters, qualities):
                break

            new_experiment = self._generate_experiment(new_parameters, old_experiment.qualities)
            experiments.append(new_experiment)

            if self._config.score_threshold >= \
               self._calculate_quality_score(new_parameters, qualities):
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
            generation_approach=self._config.type,
            optimized_qualities=qualities
        )
