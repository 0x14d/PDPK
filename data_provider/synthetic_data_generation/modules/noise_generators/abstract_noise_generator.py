"""
This module provides the abstract class `NoiseGenerator`.

To implement an own NoiseGenerator follow the steps in
`data_provider.synthetic_data_generation.config.modules.noise_generator_config.py`
"""

# pylint: disable=import-error

from abc import ABC, abstractmethod

from data_provider.synthetic_data_generation.types.experiments import GeneratedDataset

class NoiseGenerator(ABC):
    """
    Abstract class that provides functionality to add noise to a generated dataset.
    """

    def __call__(self, dataset: GeneratedDataset) -> None:
        """
        Generates noise using the `generate_noise` method.
        """
        self.generate_noise(dataset)

    @abstractmethod
    def generate_noise(self, dataset: GeneratedDataset) -> None:
        """
        Generates noise on a dataset.
        """
