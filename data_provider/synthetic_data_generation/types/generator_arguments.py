"""
This module provides type definitions regarding the arguments that get passed
to the various generator classes at initialization.
"""

# pylint: disable=no-name-in-module, too-few-public-methods, import-error

from dataclasses import dataclass

from data_provider.synthetic_data_generation.config.modules.experiment_series_generator_config \
    import ExperimentSeriesGeneratorConfig
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_generation.types.pq_function import GeneratedPQFunctions
from data_provider.synthetic_data_generation.types.pq_tuple import GeneratedPQTuples
from data_provider.knowledge_graphs.pq_relation import GeneratedPQ_Relations


@dataclass
class ExperimentGeneratorArguments:
    """
    This class contains the informations that get passed into the init method of the
    experiment-generators.
    """

    sdg_config: SdgConfig
    """Configuration of the synthetic data generator"""

    pq_functions: GeneratedPQFunctions
    """Generated pq-functions"""

    pq_tuples: GeneratedPQTuples
    """Genertated pq-tuples (correlations, expert knowledge etc.)"""


@dataclass
class ExperimentSeriesGeneratorArguments:
    """
    This class contains the informations that get passed into the generate_all_experiment_series
    method of the experiment-series.
    """

    experiment_series_config: ExperimentSeriesGeneratorConfig
    """Configuration of the experiment series"""

    sdg_config: SdgConfig
    """Configuration of the synthetic data generator"""

    pq_functions: GeneratedPQFunctions
    """Generated pq-functions"""

    pq_tuples: GeneratedPQTuples
    """Genertated pq-tuples (correlations, expert knowledge etc.)"""

@dataclass
class KnowledgeGraphGeneratorArguments:
    """
    This class contains the informations that get passed into the init method of the
    knowledge-graph-generators.
    """

    sdg_config: SdgConfig
    """Configuration of the synthetic data generator"""

    pq_functions: GeneratedPQFunctions
    """Generated pq-functions"""

    pq_tuples: GeneratedPQTuples
    """Genertated pq-tuples (correlations, expert knowledge etc.)"""
    
    pq_relations: GeneratedPQ_Relations
    """Generated pq-relations"""

@dataclass
class NoiseGeneratorArguments:
    """
    This class contains the informations that get passed into the init method of the
    noise-generators.
    """

    sdg_config: SdgConfig
    """Configuration of the synthetic data generator"""

@dataclass
class PQFunctionGeneratorArguments:
    """
    This class contains the informations that get passed into the init method of the
    pq-function-generators.
    """

    sdg_config: SdgConfig
    """Configuration of the synthetic data generator"""

    pq_tuples: GeneratedPQTuples
    """Genertated pq-tuples (correlations, expert knowledge etc.)"""

@dataclass
class PQTupleGeneratorArguments:
    """
    This class contains the informations that get passed into the init method of the
    pq-tuple-generators.
    """

    sdg_config: SdgConfig
    """Configuration of the synthetic data generator"""
