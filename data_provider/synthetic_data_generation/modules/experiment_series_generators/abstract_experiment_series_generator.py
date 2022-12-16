"""
This module provides the abstract class `ExperimentSeries`

To implement an own ExperimentSeries follow the steps in
`data_provider.synthetic_data_generation.config.modules.experiment_series_generator_config.py`
"""

# pylint: disable=import-error

from abc import ABC, abstractmethod
from itertools import product
from typing import Dict, List, Optional
import numpy as np
from numpy.random import default_rng as rng
from numpy.random import Generator

from data_provider.synthetic_data_generation.config.basic_configs.quality_config import Quality
from data_provider.synthetic_data_generation.config.modules.experiment_series_generator_config \
    import ExperimentSeriesGeneratorInitialQualityRating as InitialQualityRating
from data_provider.synthetic_data_generation.config.modules. \
    experiment_series_generator_config import AbstractExperimentSeriesGeneratorConfig
from data_provider.synthetic_data_generation.config.modules.experiment_series_generator_config \
    import ExperimentSeriesGeneratorQualityCalculationMethod as QualityCalculationMethod
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_generation.types.experiments \
    import GeneratedExperiment, GeneratedExperimentSeries
from data_provider.synthetic_data_generation.types.generator_arguments \
    import ExperimentSeriesGeneratorArguments
from data_provider.synthetic_data_generation.types.pq_function import GeneratedPQFunctions
from data_provider.synthetic_data_generation.types.pq_tuple import GeneratedPQTuples

class ExperimentSeriesGenerator(ABC):
    """
    Abstract class that provides functionality to generate experiment series.
    """

    _config: AbstractExperimentSeriesGeneratorConfig
    _sdg_config: SdgConfig
    _pq_tuples: GeneratedPQTuples
    _pq_functions: GeneratedPQFunctions
    _rng: Generator

    def __init__(self, args: ExperimentSeriesGeneratorArguments) -> None:
        self._config = args.experiment_series_config
        self._sdg_config = args.sdg_config
        self._pq_functions = args.pq_functions
        self._pq_tuples = args.pq_tuples
        self._rng = rng(self._config.seed)

    def __call__(self) -> List[GeneratedExperimentSeries]:
        return self.generate_all_experiment_series()

    @abstractmethod
    def generate_all_experiment_series(self) -> List[GeneratedExperimentSeries]:
        """
        Generates multiple experiment series.

        Returns:
            List of generated experiment series
        """

    def _generate_experiment(
            self,
            parameters: Dict[str, float],
            qualities: Dict[str, float]
        ) -> GeneratedExperiment:
        """
        Generates an experiment from the given parameter and quality values.

        The ratings for all qualities will be calculated by using the configured method.

        Parameters:
            parameters (dict): Parameter values
            qualities (dict): Quality ratings. All qualities that don't have a
                correct value will be calculated again.

        Returns:
            Generated experiment
        """
        parameters = parameters.copy()
        qualities = qualities.copy()

        for quality in qualities:
            rating = self._calculate_quality_rating(parameters, quality)
            if rating is not None:
                qualities[quality] = rating

        return GeneratedExperiment(
            parameters=parameters,
            qualities=qualities
        )

    def _calculate_quality_rating(
        self,
        parameters: Dict[str, float],
        quality: str
    ) -> Optional[float]:
        """
        Calculates the rating of a quality for the given parameters
        using the configured calculation method.

        Parameters:
            parameters (dict): Dict containing all parameter values
            quality (str): Quality for which the rating is calculated

        Returns:
            Rating of the quality
        """
        affecting_parameters = self._pq_tuples.get_parameters_affecting_quality(quality)
        if len(affecting_parameters) == 0:
            return None

        quality_ratings = []
        for parameter in affecting_parameters:
            pq_function = self._pq_functions.pq_functions[(parameter, quality)]
            quality_ratings.append(pq_function(parameters[parameter]))

        if self._config.quality_calculation_method == QualityCalculationMethod.MEAN:
            rating =  np.mean(quality_ratings)
        elif self._config.quality_calculation_method == QualityCalculationMethod.MEDIAN:
            rating = np.median(quality_ratings)
        elif self._config.quality_calculation_method == QualityCalculationMethod.BEST:
            rating = max(quality_ratings)
        elif self._config.quality_calculation_method == QualityCalculationMethod.WORST:
            rating = min(quality_ratings)
        else:
            raise NotImplementedError(
                    f'No functionality for QualityCalculationMethod \
                        {self._config.quality_calculation_method} is given!'
                )

        return self._sdg_config.get_quality_by_name(quality).limit_quality_rating(rating)

    def _calculate_quality_score(
        self,
        parameters: Dict[str, float],
        qualities: List[str]
    ) -> float:
        """
        Calculates the score of the given parameters by averaging the optimzed quality ratings.
        The score is in range [0;1] where 0 is the best possible and 1 is the worst possible score.

        Parameters:
            parameters: Parameter values
            qualities: List of the qualities to be taken into account

        Returns:
            Score of the parameterization
        """
        score = 0
        for quality in qualities:
            rating = self._calculate_quality_rating(parameters, quality)
            config = self._sdg_config.get_quality_by_name(quality)
            score += (rating - config.min_rating) / (config.max_rating - config.min_rating)
        return score / len(qualities)

    def _generate_quality_pairs_to_optimize(self, qualities: List[str]) -> List[List[str]]:
        """
        Generates a list of quality pairs that each get optimized in the same experiment series
        by using the configured constraints.

        Parameters:
            qualities: List of all qualities

        Returns:
            List of quality pairs
        """

        pair_size = self._config.num_qualities_to_optimize_per_series

        def generate_overlapping_quality_pairs():
            """
            Generate quality pairs that have overlapping influencing parameters
            """
            nonlocal qualities
            # Find overlapping qualities
            overlapping_qualities = []
            for q_1 in qualities:
                for q_2 in qualities:
                    if q_1 == q_2:
                        continue
                    p_1 = set(self._pq_tuples.get_parameters_affecting_quality(q_1))
                    p_2 = set(self._pq_tuples.get_parameters_affecting_quality(q_2))
                    overlapping_parameters = p_1 & p_2
                    if len(overlapping_parameters) > 0:
                        overlapping_qualities.append(set((q_1, q_2)))
            quality_pairs = []

            # Generate quality pairs
            self._rng.shuffle(qualities)
            while len(qualities) >= pair_size:
                next_quality = qualities.pop()
                for other_qualities in product(qualities, repeat=pair_size - 1):
                    new_pair = set(other_qualities)
                    new_pair.add(next_quality)

                    def is_pair_valid(pair) -> bool:
                        if len(pair) != pair_size:
                            return False

                        for q_1 in pair:
                            if not any(set((q_1, q_2)) in overlapping_qualities for q_2 in pair):
                                return False
                        return True

                    if not is_pair_valid(new_pair):
                        continue

                    quality_pairs.append(list(new_pair))
                    qualities = [q for q in qualities if q not in new_pair]
                    break

            if len(quality_pairs) == 0:
                raise Exception('There are no overlapping qualities!')

            return quality_pairs

        if self._config.only_optimize_qualities_with_overlapping_parameters and \
           self._config.num_qualities_to_optimize_per_series > 1:
            return generate_overlapping_quality_pairs()

        # Generate random pairs
        quality_pairs = []
        while len(qualities) >= pair_size:
            chosen_qualities = self._rng.choice(
                a=qualities,
                size=pair_size,
                replace=False
            )
            qualities = [q for q in qualities if q not in chosen_qualities]
            quality_pairs.append(chosen_qualities)
        return quality_pairs

    def _generate_first_experiment(self, qualities_to_optimize: List[str]) -> GeneratedExperiment:
        """
        Generates the first experiment of the experiment series.

        Parameters:
            qualities_to_optimize (List): Names of the qualities that are getting optimized

        Returns:
            The generated experiment
        """
        parameters: Dict[str, float] = {}
        qualities: Dict[str, float] = self._generate_initial_qualities(qualities_to_optimize)

        for parameter in self._pq_tuples.selected_parameters:
            parameter_config = self._sdg_config.get_parameter_by_name(parameter)
            affected_qualities = self._pq_tuples.get_qualities_affected_by_paramter(parameter)

            if len(affected_qualities) == 0:
                # Random value if parameter doesn't affect any quality
                parameters[parameter] = self._rng.uniform(
                    low=parameter_config.min_value,
                    high=parameter_config.max_value
                )
                continue

            affected_qualities_to_optimize = set(affected_qualities) & set(qualities_to_optimize)
            if len(affected_qualities_to_optimize) > 0:
                # Set parameter to only guarantee the optimized quality
                affected_qualities = list(affected_qualities_to_optimize)

            p_0 = 0
            for quality in affected_qualities:
                pq_function = self._pq_functions.pq_functions[(parameter, quality)]
                p_0 += pq_function.inverse(quality=qualities[quality])
            p_0 /= len(affected_qualities)

            parameters[parameter] = p_0

        return self._generate_experiment(parameters, qualities)

    def _generate_initial_qualities(self, qualities_to_optimize: List[str]) -> Dict[str, float]:
        """
        Generates the initial quality ratings. The quality that is getting optimizedis initilized
        with the configured method. All other qualities are initialized with random values.

        Parameters:
            qualities_to_optimize (str): Names of the qualities that are getting optimized

        Returns:
            Dict containing the initial quality values
        """
        def random_quality(quality: Quality) -> float:
            """
            Returns a random quality rating for the given quality.
            """
            return self._rng.uniform(low=quality.min_rating, high=quality.max_rating)

        qualities: Dict[str, float] = {}

        for quality in self._pq_tuples.selected_qualities:
            quality_config = self._sdg_config.get_quality_by_name(quality)
            if quality in qualities_to_optimize:
                if self._config.initial_quality_rating == InitialQualityRating.WORST:
                    quality_offset = (quality_config.max_rating - quality_config.min_rating) * 0.1
                    quality_offset = self._rng.uniform(low=0, high=quality_offset)
                    qualities[quality] = quality_config.max_rating - quality_offset
                elif self._config.initial_quality_rating == InitialQualityRating.RANDOM:
                    qualities[quality] = random_quality(quality_config)
                else:
                    raise NotImplementedError(
                        f'No functionality for InitialQualityRating \
                            {self._config.initial_quality_rating} is given!'
                    )
            else:
                qualities[quality] = random_quality(quality_config)

        return qualities
