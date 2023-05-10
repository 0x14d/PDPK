"""
This module provides the class `TrialExperimentSeriesGenerator`.
"""

# pylint: disable=import-error, relative-beyond-top-level, useless-super-delegation, too-few-public-methods, too-many-statements

from typing import List, Optional

from data_provider.synthetic_data_generation.types.experiments \
    import GeneratedExperiment, GeneratedExperimentSeries
from data_provider.synthetic_data_generation.types.generator_arguments \
    import ExperimentSeriesGeneratorArguments
from data_provider.synthetic_data_generation.config.modules.experiment_series_generator_config \
    import TrialExperimentSeriesGeneratorConfig
from data_provider.synthetic_data_generation.config.modules.experiment_series_generator_config \
    import ExperimentSeriesGeneratorTrialParameterAdjustmentMethod as ParameterAdjustmentMethod
from .abstract_experiment_series_generator import ExperimentSeriesGenerator

class TrialExperimentSeriesGenerator(ExperimentSeriesGenerator):
    """
    Class that provides functionality to generate experiment series.

    Each experiment series optimizes a set of qualities by trying out different parameter
    adjustments. It starts with a random initialized experiment and in each step some parameters
    are adjusted in order to change the resulting qualities. If the qualities get better the same
    parameters are further adjusted. Otherwise new parameters are tested.
    """
    _config: TrialExperimentSeriesGeneratorConfig

    def __init__(self, args: ExperimentSeriesGeneratorArguments) -> None:
        super().__init__(args)

        if self._config.override_expert_knowledge:
            self._pq_tuples.expert_knowledge = []

    def generate_all_experiment_series(self) -> List[GeneratedExperimentSeries]:
        """
        Generates an experiment series for every quality with testing parametrizations.

        Returns:
            List of generated experiment series
        """
        qualities: List[str] = self._pq_tuples.selected_qualities.copy()
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
        best_experiment = experiments[-1]
        best_experiment_score = self._calculate_quality_score(best_experiment.parameters, qualities)

        def adjust_parameter(parameter: str, parameter_value: float, direction: int) -> float:
            """
            Returns the adjusted parameter value by using the configured adjustment method
            """
            method: ParameterAdjustmentMethod = self._config.parameter_adjustment_method
            parameter_config = self._sdg_config.get_parameter_by_name(parameter)
            p_definition_range_size = parameter_config.max_value - parameter_config.min_value
            p_definition_range_step = p_definition_range_size * self._config.parameter_adjustment
            p_after_d_range = parameter_value + p_definition_range_step * direction

            if method == ParameterAdjustmentMethod.PERCENTAGE_OF_DEFINTION_RANGE_THEN_CURRENT_VALUE:
                if parameter_config.min_value <= p_after_d_range <= parameter_config.max_value:
                    method = ParameterAdjustmentMethod.PERCENTAGE_OF_DEFINTION_RANGE
                else:
                    method = ParameterAdjustmentMethod.PERCENTAGE_OF_CURENT_VALUE

            if method == ParameterAdjustmentMethod.PERCENTAGE_OF_DEFINTION_RANGE:
                return parameter_config.limit_parameter_value(p_after_d_range)
            if method == ParameterAdjustmentMethod.PERCENTAGE_OF_CURENT_VALUE:
                return parameter_config.limit_parameter_value(
                    parameter_value * (1 + self._config.parameter_adjustment * direction)
                )
            raise NotImplementedError('No implementation for parameter adjustment method {method}!')

        def add_expert_knowledge(
            old_experiment: GeneratedExperiment,
            new_experiment: GeneratedExperiment,
            parameter: str
        ):
            for quality in self._pq_tuples.selected_qualities:
                if old_experiment.qualities[quality] != new_experiment.qualities[quality] \
                   and (parameter, quality) not in self._pq_tuples.expert_knowledge:
                    self._pq_tuples.expert_knowledge.append((parameter, quality))

        def explore_direction(
            parameter: str,
            base_experiment: GeneratedExperiment,
            direction: int
        ) -> Optional[bool]:
            """
            Explores the parameter in a specific direction
            until the scores doesn't imporove anymore.
            """
            nonlocal best_experiment, best_experiment_score
            last_experiment = base_experiment
            last_experiment_score = self._calculate_quality_score(
                last_experiment.parameters, qualities
            )
            new_experiment_score = last_experiment_score

            new_experiments: List[GeneratedExperiment] = []
            while True:
                p_old = last_experiment.parameters[parameter]
                p_new = adjust_parameter(parameter, p_old, direction)

                if p_old == p_new:
                    # End if parameter has same value as before
                    break

                new_parameters = last_experiment.parameters.copy()
                new_parameters[parameter] = p_new
                new_experiment = self._generate_experiment(
                    new_parameters, last_experiment.qualities, qualities, last_experiment
                )
                new_experiments.append(new_experiment)

                new_experiment_score = self._calculate_quality_score(new_parameters, qualities)

                if self._config.override_expert_knowledge:
                    add_expert_knowledge(last_experiment, new_experiment, parameter)

                if new_experiment_score >= last_experiment_score:
                    # End if score didn't improve
                    break

                if new_experiment_score < best_experiment_score:
                    best_experiment = new_experiment
                    best_experiment_score = new_experiment_score

                if new_experiment_score <= self._config.score_threshold:
                    # End if score is good enough
                    break

                last_experiment = new_experiment
                last_experiment_score = new_experiment_score

            experiments.extend(new_experiments)
            if len(new_experiments) > 1:
                # Base experiment was imporved
                return True
            if new_experiment_score == last_experiment_score:
                # Score wasn't influenced
                return None
            # Score was disimproved
            return False

        adjusted_parameters = self._rng.permutation(self._pq_tuples.selected_parameters).tolist()
        for parameter in adjusted_parameters:
            directions = self._rng.permutation([1, -1])
            base_experiment = best_experiment
            for direction in directions:
                exploration_status = explore_direction(parameter, base_experiment, direction)
                if exploration_status is True and self._config.only_explore_one_improving_direction:
                    break
                if exploration_status is None:
                    break
            if len(experiments) >= self._config.max_trial_length:
                break

        return GeneratedExperimentSeries(
            experiments=experiments,
            generation_approach=self._config.type
        )
