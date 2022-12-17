"""
This module provides everything that is needed for the configuration of the
experiment-series-generator.

To add a new function follow these steps:
1. Create a new class that inherits from `ExperimentSeriesGenerator` and implements the required
methods
(in a new file in the `data_provider/synthetic_data_generation/modules/experiment_series_generators`
directory)
2. Add a type name for the new function to the `ExperimentSeriesGeneratorType` enum
3. Create a new config class for the new function (in this module). It should inherit from
`AbstractExperimentSeriesGeneratorConfig` and have an attribute `type` of datatype
`Literal[ExperimentSeriesGeneratorType.<New enum member (see step 2)>]`.
4. Add a new if case to the `get_configuration` method of `ExperimentSeriesGeneratorType` that
returns the new generator configuration if `self == ExperimentSeriesGeneratorType.<new enum member>`
5. Add the new config class to the `ExperimentSeriesGeneratorConfig` type by adding the new class to
the union (`Union[ExperimentSeriesGeneratorType, ..., <new config class>]`)
"""

# pylint: disable=no-name-in-module, import-error, import-outside-toplevel, no-self-argument, too-few-public-methods, no-self-use, missing-class-docstring

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel

class ExperimentSeriesGeneratorType(str, Enum):
    """
    Enum that contains the key of every available experiment-series-generator. It's used in the
    configuration to specify what series are used.

    When adding a new generator a new enum member must be added!
    """
    EXPERT_KNOWLEDGE = 'expert_knowledge'
    """Type of the `ExpertKnowledgeExperimentSeriesGenerator`"""

    TRIAL = 'trial'
    """Type of the `TrialExperimentSeriesGenerator`"""

    def get_configuration(self) -> AbstractExperimentSeriesGeneratorConfig:
        """
        Creates the matching configuration of the experiment-series-generator-type.

        Returns:
            experiment-series-generator configuaration with its default values
        """
        if self == ExperimentSeriesGeneratorType.EXPERT_KNOWLEDGE:
            return ExpertKnowledgeExperimentSeriesGeneratorConfig()
        if self == ExperimentSeriesGeneratorType.TRIAL:
            return TrialExperimentSeriesGeneratorConfig()
        raise NotImplementedError(f'Missing implementation for type {self}!')

class ExperimentSeriesGeneratorQualityCalculationMethod(str, Enum):
    """
    Enum that contains all methods to calculate the quality rating from multiple influencing
    parameters.
    """
    MEAN = "mean"
    """Calculate the mean of all quality ratings"""

    MEDIAN = "median"
    """Calculate the median of all quality ratings"""

    BEST = "best"
    """Calculate the maximum of all quality ratings"""

    WORST = "worst"
    "Calculate the minimum of all quality ratings"

class ExperimentSeriesGeneratorInitialQualityRating(str, Enum):
    """
    Enum that contains all methods to initialize the quality rating that is getting optimized.
    """
    WORST = "worst"
    """Initialize it with the worst possible rating"""

    RANDOM = "random"
    """Initialize it with a random rating"""

class ExperimentSeriesGeneratorTrialParameterAdjustmentMethod(str, Enum):
    """
    Enum that contains all methods to adjust a parameter value in a trial experiment series
    """
    PERCENTAGE_OF_DEFINTION_RANGE = "definition_range"
    """Adjust the parameter by a percentage of the parmeters definintion range"""

    PERCENTAGE_OF_CURENT_VALUE = "current_value"
    """
    Adjust the parameter by a percentage of the parmeters current value.
    This method needs longer to converge.
    """

    PERCENTAGE_OF_DEFINTION_RANGE_THEN_CURRENT_VALUE = "definition_range_then_current_value"
    """
    Adjust the parameter by a percentage of the parmeters definintion range until the parameter
    value would exceed the definition range then adjust it by a percentage of the parmeters
    current value.
    """

class AbstractExperimentSeriesGeneratorConfig(BaseModel, ABC):
    """
    Abstract class for the experiment-series-generator configurations.

    When adding a new generator the associated configuration class must inherit from this class!
    """

    initial_quality_rating: ExperimentSeriesGeneratorInitialQualityRating \
        = ExperimentSeriesGeneratorInitialQualityRating.WORST
    """Determines how the qualities that get optimized are initialized"""

    num_qualities_to_optimize_per_series: int = 1
    """Number of qualities that are getting optimized per experiment series"""

    only_optimize_qualities_with_overlapping_parameters: bool = False
    """
    If True only qualities with at least one overlapping influencing parameter
    are getting optimized in an experiment series.
    If False the optimized qualities are chosen randomly.
    """

    quality_calculation_method: ExperimentSeriesGeneratorQualityCalculationMethod = \
        ExperimentSeriesGeneratorQualityCalculationMethod.MEAN
    """Determines how the quality is calculated from multiple parameters"""

    score_threshold: float = 0.05
    """
    Threshold that determines when to stop the parameter opimization.
    The Score is in range [0;1] where 0 is the best and 1 the worst possible score.
    """

    seed: int = 42
    """Seed of the random number generator"""

    @abstractmethod
    def get_generator_class(self) -> Any:
        """
        Returns the experiment-series-generator class this config is about.

        IMPORTANT: Import the returned class in this method and not outside of it!
        """

class ExpertKnowledgeExperimentSeriesGeneratorConfig(AbstractExperimentSeriesGeneratorConfig):
    """
    Configuration of the `ExpertKnowledgeExperimentSeriesGenerator`
    """
    type: Literal[ExperimentSeriesGeneratorType.EXPERT_KNOWLEDGE] \
        = ExperimentSeriesGeneratorType.EXPERT_KNOWLEDGE

    max_series_size: int = 15
    """Maximum number of experiments a series can have"""

    def get_generator_class(self):
        from data_provider.synthetic_data_generation.modules.experiment_series_generators. \
        expert_knowledge_experiment_series_generator import ExpertKnowledgeExperimentSeriesGenerator
        return ExpertKnowledgeExperimentSeriesGenerator

class TrialExperimentSeriesGeneratorConfig(AbstractExperimentSeriesGeneratorConfig):
    """
    Configuration of the `TrialExperimentSeriesGenerator`
    """
    type: Literal[ExperimentSeriesGeneratorType.TRIAL] \
        = ExperimentSeriesGeneratorType.TRIAL

    max_trial_length: int = 25
    """
    Number of experiements after the trial is stopped. If a positive influencing parameter is found
    the max length can be exceeded.
    """

    only_explore_one_improving_direction: bool = True
    """
    If True the parameter exploration will stop if an improving direction was found.
    If False the paramater will be explored in the other direction as well.
    """

    parameter_adjustment_method: ExperimentSeriesGeneratorTrialParameterAdjustmentMethod \
        = ExperimentSeriesGeneratorTrialParameterAdjustmentMethod.PERCENTAGE_OF_DEFINTION_RANGE
    """Determines how a parameter value gets adjusted in an experiment series"""

    parameter_adjustment: float = 0.1
    """Adjustment value used by the `parameter_adjustment_method`"""

    def get_generator_class(self):
        from data_provider.synthetic_data_generation.modules.experiment_series_generators. \
        trial_experiment_series_generator import TrialExperimentSeriesGenerator
        return TrialExperimentSeriesGenerator


ExperimentSeriesGeneratorConfig = Union[
                                    ExpertKnowledgeExperimentSeriesGeneratorConfig,
                                    TrialExperimentSeriesGeneratorConfig,
                                    ExperimentSeriesGeneratorType
                                ]
"""Type alias that contains all available experiment-series-generator configuration classes."""


DEFAULT_EXPERIMENT_SERIES_GENERATOR_CONFIG: ExperimentSeriesGeneratorType \
    = ExperimentSeriesGeneratorType.EXPERT_KNOWLEDGE
"""Default experiment-series-generator type that is used if no configuration is provided."""


def parse_experiment_series_generator_config(
    config: Optional[ExperimentSeriesGeneratorConfig]
) -> AbstractExperimentSeriesGeneratorConfig:
    """
    Parses a experiment-series-generator configuration to its default format.

    If the configuration is None the default configuration is used.

    Parameters:
        config (ExperimentSeriesGeneratorConfig | None): config that should be parsed

    Returns:
        config in the default format (AbstractExperimentSeriesGeneratorConfig)
    """
    if config is None:
        config = DEFAULT_EXPERIMENT_SERIES_GENERATOR_CONFIG
    if isinstance(config, ExperimentSeriesGeneratorType):
        return config.get_configuration()
    return config
