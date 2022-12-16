"""
This module provides everything that is needed for the configuration of the process qualities.

To add a new type of quality configuration follow these steps:
1. Add the new configuration type to the `QualityConfig` by adding the type to the union
(`Union[DefaultQualityConfig, ..., <new type>]`).
2. Modify the `parse_quality_config_to_default_config` function to support the parsing of
the new type.
"""

# pylint: disable=no-name-in-module, too-few-public-methods, no-self-argument, no-self-use

from typing import List, Optional, Union
from pydantic import BaseModel, validator
from numpy.typing import ArrayLike

class QualityProperties(BaseModel):
    """
    Class that contains all properties of a process quality.
    """

    max_rating: int = 10
    """Maximum rating value the quality can have"""

    min_rating: int = 0
    """Minimum rating value the quality can have"""

    @validator('min_rating')
    def min_rating_mustnt_be_negative(cls, min_rating):
        """
        Checks if `min_rating` >= 0.
        """
        if min_rating < 0:
            raise ValueError('must be >= 0')
        return min_rating

    @validator('min_rating')
    def min_rating_must_be_less_than_max_rating(cls, min_rating, values):
        """
        Checks if `min_rating` < `max_rating`.
        """
        if 'max_rating' in values and min_rating >= values['max_rating']:
            raise ValueError('must be < max_rating')
        return min_rating

    def limit_quality_rating(self, quality: float) -> float:
        """
        Limits a quality rating to fit in the definition range of the quality.

        Parameters:
            quality (float): Quality rating

        Returns:
            Quality rating in range [self.min_rating, self.max_rating]
        """
        return max(self.min_rating, min(quality, self.max_rating))

    def limit_quality_ratings(self, qualities: ArrayLike) -> ArrayLike:
        """
        Limits a list quality rating to fit in the definition range of the quality.

        Parameters:
            qualities (ArrayLike): Quality ratings

        Returns:
            Quality ratings in range [self.min_rating, self.max_rating]
        """
        return [max(self.min_rating, min(quality, self.max_rating)) for quality in qualities]

class Quality(QualityProperties):
    """
    Class that contains all information about a process quality.

    While `QualityProperties` contains information about qualities that can be shared
    between multiple qualities, `Quality` contains the information for one specific quality.
    """
    name: str
    """Name of the quality"""

class QualitiesWithSharedPropertiesConfig(QualityProperties):
    """
    Quality configuraton that contains a list of qualities that can either be `Quality` objects
    or `str` (quality name).

    Qualities with no properties (only name) gain the shared quality properties
    that are provided in this configuration.
    """

    qualities: List[Union[str, Quality]]
    """List of qualities / quality names"""


DefaultQualityConfig = List[Quality]
"""
Type alias for the quality configuration type that is used after loading the configuration.
All other configuration types are getting parsed to this type upon loading.
"""

QualityConfig = Union[
    DefaultQualityConfig,
    List[Union[Quality, str]],
    QualitiesWithSharedPropertiesConfig
]
"""
Type alias that contains all available quality configuration type.

Supported configuration types:
- List of
    - `Quality` or
    - `str` representing the quality name
- Qualities with shared properties:
    - List of
        - `Quality` or
        - `str` representing the quality name
    - Quality properties that are shared between all qualities which are defined by their names
"""


def parse_quality_config_to_default_config(config: QualityConfig) -> DefaultQualityConfig:
    """
    Function that parses a `QualityConfig` to the `DefaultQualityConfig`.

    When adding a new quality config type this function has to be modified.

    Parameters:
        config (QualityConfig): The quality config that should be parsed.

    Returns:
        default_config (DefaultQualityConfig): Parsed config
    """
    def parse_quality_list(
        qualities: List[Union[str, Quality]],
        properties: Optional[dict] = None
    ) -> List[Quality]:
        """
        Converts a list of `str` and `Quality` to a list of `Quality`.

        Parameters:
            qualities (list[str | Quality]): List of qualities / quality names
            properties (dict): Optional properties that are used to convert a quality name to a
                quality. If not defined the default properties are used.
        """
        parsed_qualities: List[Quality] = []
        for qual in qualities:
            if isinstance(qual, str):
                # Case: qual is quality name
                parsed_qualities.append(Quality(name=qual, **(properties if properties else {})))
            else:
                # Case: qual is Quality object
                parsed_qualities.append(qual)
        return parsed_qualities

    if isinstance(config, QualitiesWithSharedPropertiesConfig):
        properties = {k: v for k, v in config.dict().items() if k != 'qualities'}
        return parse_quality_list(config.qualities, properties)

    return parse_quality_list(config)
