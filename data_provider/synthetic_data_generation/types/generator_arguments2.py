"""
This module provides type definitions regarding the arguments that get passed
to the various generator classes at initialization.

This second module is used to prevent circular imports.
"""

# pylint: disable=import-error

from dataclasses import dataclass

from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_generation.types.pq_tuple import GeneratedPQTuples


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
