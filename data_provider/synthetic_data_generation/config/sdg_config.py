"""
This module provides the SdgConfig class that manages the configuration of the synthetic data
generator.
"""

# pylint: disable=no-name-in-module, no-self-argument, no-self-use, too-few-public-methods, import-error

from __future__ import annotations

from typing import Optional, Union
from pydantic import BaseModel, root_validator

from data_provider.knowledge_graphs.config.knowledge_graph_generator_config import \
    KnowledgeGraphGeneratorConfig, parse_knowledge_graph_generator_config
from .modules.noise_generator_config import NoiseGeneratorConfig, parse_noise_generator_config
from .modules.experiment_generator_config import \
    ExperimentGeneratorConfig, parse_experiment_generator_config
from .modules.pq_function_generator_config import \
    PQFunctionGeneratorConfig, parse_pq_function_generator_config
from .modules.pq_tuple_generator_config import \
    PQTupleGeneratorConfig, parse_pq_tuple_generator_config
from .basic_configs.parameter_config import \
    Parameter, ParameterConfig, parse_parameter_config_to_default_config
from .basic_configs.quality_config import \
    Quality, QualityConfig, parse_quality_config_to_default_config

class SdgConfig(BaseModel):
    """
    This class represents the configuration of the synthetic data generator.

    It uses the pydantic `BaseModel` class to provide additional functionality like
    json serialization.
    """

    dataset_generator: Optional[ExperimentGeneratorConfig]
    """
    Provides information about the experiment-generator.
    It determines what generator is used and how it's configured.

    For further information look at
    `data_provider.synthetic_data_generation.config.modules.dataset_generator_config.py`
    """

    knowledge_graph_generator: Optional[KnowledgeGraphGeneratorConfig]
    """
    Provides information about the knowledge-graph-generator.
    It determines what generator is used and how it's configured.

    For further information look at
    `data_provider.synthetic_data_generation.config.modules.knowledge_graph_generator_config.py`
    """

    noise_generator: Optional[NoiseGeneratorConfig]
    """
    Provides information about the noise-generator.
    It determines what generator is used and how it's configured.

    For further information look at
    `data_provider.synthetic_data_generation.config.modules.noise_generator_config.py`
    """

    parameters: ParameterConfig
    """
    Provides information about all parameters and their properties.

    For further information look at
    `data_provider.synthetic_data_generation.config.basic_configs.parameter_config.py`
    """

    pq_function_generator: Optional[PQFunctionGeneratorConfig]
    """
    Provides information about the pq-function-generator.
    It determines what generator is used and how it's configured.

    For further information look at
    `data_provider.synthetic_data_generation.config.modules.pq_function_generator_config.py`
    """

    pq_tuple_generator: Optional[PQTupleGeneratorConfig]
    """
    Provides information about the pq-tuple-generator.
    It determines what generator is used and how it's configured.

    For further information look at
    `data_provider.synthetic_data_generation.config.modules.pq_tuple_generator_config.py`
    """

    qualities: QualityConfig
    """
    Provides information about all qualities and their properties.

    For further information look at
    `data_provider.synthetic_data_generation.config.basic_configs.quality_config.py`
    """

    @root_validator
    def parse_configs(cls, config):
        """
        Parses all configurations to a default format
        """
        config['parameters'] = parse_parameter_config_to_default_config(config['parameters'])
        config['qualities'] = parse_quality_config_to_default_config(config['qualities'])
        config['dataset_generator'] = parse_experiment_generator_config(
                                            config['dataset_generator']
                                         )
        config['knowledge_graph_generator'] = parse_knowledge_graph_generator_config(
                                                config['knowledge_graph_generator']
                                              )
        config['noise_generator'] = parse_noise_generator_config(config['noise_generator'])
        config['pq_function_generator'] = parse_pq_function_generator_config(
                                            config['pq_function_generator']
                                          )
        config['pq_tuple_generator'] = parse_pq_tuple_generator_config(config['pq_tuple_generator'])
        return config

    def get_quality_by_name(self, quality: str) -> Quality:
        """
        Returns the quality with the matching name
        """
        return [q for q in self.qualities if q.name == quality][0]

    def get_parameter_by_name(self, parameter: str) -> Parameter:
        """
        Returns the parameter with the matching name
        """
        return [p for p in self.parameters if p.name == parameter][0]

    @staticmethod
    def create_config(config: Union[str, dict]) -> SdgConfig:
        """
        Creates a `SdgConfig` object from either a dict containing the configurations
        or a str containing the path to the configuration file.

        Parameters:
            config (str | dict): information about the configuration

        Returns:
            Imported / parsed configuration
        """
        if isinstance(config, str):
            return SdgConfig.parse_file(config)
        if isinstance(config, dict):
            return SdgConfig.parse_obj(config)
        raise ValueError('config must be a str or dict')
