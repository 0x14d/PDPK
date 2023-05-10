"""
This module provides everything that is needed for the configuration of the
knowledge-graph-generators.

To add a new generator follow these steps:
1. Create a new class that inherits from `KnowledgeGraphGenerator` and implements the required
methods
(in a new file in the `data_provider/synthetic_data_generation/modules/knowledge_graph_generators`
directory)
2. Add a type name for the new generator to the `KnowledgeGraphGeneratorType` enum
3. Create a new config class for the new generator (in this module). It should inherit from
`AbstractKnowledgeGraphGeneratorConfig` and have an attribute `type` of datatype
`Literal[KnowledgeGraphGeneratorType.<New enum member (see step 2)>]`.
4. Add a new if case to the `get_configuration` method of `KnowledgeGraphGeneratorType` that returns
the new generator configuration if `self == KnowledgeGraphGeneratorType.<new enum member>`
5. Add the new config class to the `KnowledgeGraphGeneratorConfig` type by adding the new class to
the union (`Union[KnowledgeGraphGeneratorType, ..., <new config class>]`)
"""

# pylint: disable=no-name-in-module, import-error, import-outside-toplevel, too-few-public-methods

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field
from typing_extensions import Annotated


class KnowledgeGraphGeneratorType(str, Enum):
    """
    Enum that contains the key of every available knowledge-graph-generator.
    It's used in the configuration to specify what generator is used.

    When adding a new generator a new enum member must be added!
    """
    UNQUANTIFIED = 'unquantified'
    BASIC = 'basic'
    QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT = 'quantified_parameters_without_shortcut'
    QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT2 = 'quantified_parameters_without_shortcut2'
    QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT3 = 'quantified_parameters_without_shortcut3'
    QUANTIFIED_PARAMETERS_WITH_SHORTCUT = 'quantified_parameters_with_shortcut'
    QUANTIFIED_PARAMETERS_WITH_SHORTCUT2 = 'quantified_parameters_with_shortcut2'
    QUANTIFIED_PARAMETERS_WITH_SHORTCUT3 = 'quantified_parameters_with_shortcut3'
    QUANTIFIED_PARAMETERS_WITH_LITERAL = 'quantified_parameters_with_literal'
    QUANTIFIED_PARAMETERS_W3 = 'quantified_parameters_w3'
    QUANTIFIED_PARAMETERS_W3_WITH_LITERAL = 'quantified_parameters_w3_with_literal'
    QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT = 'quantified_conditions_without_shortcut'
    QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT2 = 'quantified_conditions_without_shortcut2'
    QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT3 = 'quantified_conditions_without_shortcut3'
    QUANTIFIED_CONDITIONS_WITH_SHORTCUT = 'quantified_conditions_with_shortcut'
    QUANTIFIED_CONDITIONS_WITH_SHORTCUT2 = 'quantified_conditions_with_shortcut2'
    QUANTIFIED_CONDITIONS_WITH_SHORTCUT3 = 'quantified_conditions_with_shortcut3'
    QUANTIFIED_CONDITIONS_WITH_LITERAL = 'quantified_conditions_with_literal'
    QUANTIFIED_CONDITIONS_W3 = 'quantified_conditions_w3'
    QUANTIFIED_CONDITIONS_W3_WITH_LITERAL = 'quantified_conditions_w3_with_literal'

    def get_configuration(self) -> AbstractKnowledgeGraphGeneratorConfig:
        """
        Creates the matching configuration of the knowledge-graph-generator-type.

        Returns:
            knowledge-graph-generator configuaration with its default values
        """
        if self == KnowledgeGraphGeneratorType.UNQUANTIFIED:
            return UnquantifiedKnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.BASIC:
            return BasicKnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_LITERAL:
            return QuantifiedParametersWithLiteralKnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT:
            return QuantifiedParametersWithShortcutKnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT2:
            return QuantifiedParametersWithShortcut2KnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT3:
            return QuantifiedParametersWithShortcut3KnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT:
            return QuantifiedParametersWithoutShortcutKnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT2:
            return QuantifiedParametersWithoutShortcut2KnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT3:
            return QuantifiedParametersWithoutShortcut3KnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_W3:
            return QuantifiedParametersW3KnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_W3_WITH_LITERAL:
            return QuantifiedParametersW3WithLiteralKnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT:
            return QuantifiedConditionsWithoutShortcutKnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT2:
            return QuantifiedConditionsWithoutShortcut2KnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT3:
            return QuantifiedConditionsWithoutShortcut3KnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT:
            return QuantifiedConditionsWithShortcutKnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT2:
            return QuantifiedConditionsWithShortcut2KnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT3:
            return QuantifiedConditionsWithShortcut3KnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_LITERAL:
            return QuantifiedConditionsWithLiteralKnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_W3:
            return QuantifiedParametersW3KnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_W3:
            return QuantifiedConditionsW3KnowledgeGraphGeneratorConfig()
        if self == KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_W3_WITH_LITERAL:
            return QuantifiedConditionsW3WithLiteralKnowledgeGraphGeneratorConfig()
        raise NotImplementedError(f'Missing implementation for type {self}!')

    @property
    def latex_label(self) -> str:
        """Latex label of the knowledge graph type"""
        labels = {
            KnowledgeGraphGeneratorType.UNQUANTIFIED: r"$r_{\eta}$",
            KnowledgeGraphGeneratorType.BASIC: r"$r_{\hat{\rho}, rel}$",
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT: r"$r_{\hat{\rho}, ch,e}$",
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT: r"$r_{\hat{\rho}, ch, e, \eta}$",
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_LITERAL: r"$r_{\hat{\rho}, ch, l, \eta}$",
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_W3: r"$r_{\hat{\rho}, rei, e, \eta}$",
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_W3_WITH_LITERAL: r"$r_{\hat{\rho}, rei, l, \eta}$",
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT: r"$r_{\hat{\mathrm{o}},\hat{\rho}, ch, e}$",
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT: r"$r_{\hat{\mathrm{o}},\hat{\rho}, ch, e, \eta}$",
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_LITERAL: r"$r_{\hat{\mathrm{o}},\hat{\rho}, ch, l, \eta}$",
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_W3: r"$r_{\hat{\mathrm{o}},\hat{\rho}, rei, e, \eta}$",
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_W3_WITH_LITERAL: r"$r_{\hat{\mathrm{o}},\hat{\rho}, rei, l, \eta}$",
        }
        labels[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT2] = labels[
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT] + ' 2'
        labels[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT3] = labels[
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT] + ' 3'
        labels[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT2] = labels[
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT] + ' 2'
        labels[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT3] = labels[
            KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT] + ' 3'
        labels[KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT2] = labels[
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT] + ' 2'
        labels[KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT3] = labels[
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT] + ' 3'
        labels[KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT2] = labels[
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT] + ' 2'
        labels[KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT3] = labels[
            KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT] + ' 3'
        return labels.get(self, self.value)


class AbstractKnowledgeGraphGeneratorConfig(BaseModel, ABC):
    """
    Abstract class for the knowledge-graph-generator configurations.

    When adding a new generator the associated configuration class must inherit from this class!
    """

    @abstractmethod
    def get_generator_class(self) -> Any:
        """
        Returns the class of the knowledge-graph-generator this config is about.

        IMPORTANT: Import the returned class in this method and not outside of it!
        """


class BasicKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `BasicKnowledgeGraphGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.BASIC] = KnowledgeGraphGeneratorType.BASIC

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.basic_knowledge_graph_generator import BasicKnowledgeGraphGenerator
        return BasicKnowledgeGraphGenerator


class QuantifiedParametersWithoutShortcutKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `QuantifiedParametersWithoutShortcutKnowledgeGraphGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT] = KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.quantified_parameters_without_shortcut import QuantifiedParametersWithoutShortcut
        return QuantifiedParametersWithoutShortcut


class QuantifiedParametersWithoutShortcut2KnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `QuantifiedParametersWithoutShortcut2KnowledgeGraphGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT2] = KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT2

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.quantified_parameters_without_shortcut2 import QuantifiedParametersWithoutShortcut2
        return QuantifiedParametersWithoutShortcut2


class QuantifiedParametersWithoutShortcut3KnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `QuantifiedParametersWithoutShortcut3KnowledgeGraphGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT3] = KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITHOUT_SHORTCUT3

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.quantified_parameters_without_shortcut3 import QuantifiedParametersWithoutShortcut3
        return QuantifiedParametersWithoutShortcut3


class QuantifiedParametersWithShortcutKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `QuantifiedParametersWithShortcutKnowledgeGraphGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT] = KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.quantified_parameters_with_shortcut import QuantifiedParametersWithShortcut
        return QuantifiedParametersWithShortcut


class QuantifiedParametersWithShortcut2KnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `QuantifiedParametersWithShortcut2KnowledgeGraphGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT2] = KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT2

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.quantified_parameters_with_shortcut2 import QuantifiedParametersWithShortcut2
        return QuantifiedParametersWithShortcut2


class QuantifiedParametersWithShortcut3KnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `QuantifiedParametersWithShortcut3KnowledgeGraphGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT3] = KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_SHORTCUT3

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.quantified_parameters_with_shortcut3 import QuantifiedParametersWithShortcut3
        return QuantifiedParametersWithShortcut3


class QuantifiedParametersWithLiteralKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `QuantifiedParametersWithLiteralKnowledgeGraphGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_LITERAL] = KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_WITH_LITERAL

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.quantified_parameters_with_literal import QuantifiedParametersWithLiteral
        return QuantifiedParametersWithLiteral


class QuantifiedParametersW3KnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `QuantifiedParametersW3KnowledgeGraphGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_W3] = KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_W3

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.quantified_parameters_w3 import QuantifiedParametersW3
        return QuantifiedParametersW3


class QuantifiedParametersW3WithLiteralKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `QuantifiedParametersW3WithLiteralKnowledgeGraphGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_W3_WITH_LITERAL] = KnowledgeGraphGeneratorType.QUANTIFIED_PARAMETERS_W3_WITH_LITERAL

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.quantified_parameters_w3 import QuantifiedParametersW3
        return QuantifiedParametersW3


class UnquantifiedKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the `UnquantifiedKGGenerator`
    """
    type: Literal[KnowledgeGraphGeneratorType.UNQUANTIFIED] = KnowledgeGraphGeneratorType.UNQUANTIFIED

    knowledge_share: float = 1.0
    """Proportion of the expert knowledge that is included into the knowledge graph"""

    seed: int = 42
    """Seed of the random number generator"""

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.unqantified_knowledge_graph_generator import UnquantifiedKnowledgeGraphGenerator
        return UnquantifiedKnowledgeGraphGenerator


class QuantifiedConditionsWithoutShortcutKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the 'QuantifiedConditionsWithoutShortcutGenerarator'
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT] = KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT

    seed: int = 42

    knowledge_share: float = 1.0

    number_of_bins: int = 5

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.qc_without_shortcut import QcWithoutShortcut
        return QcWithoutShortcut


class QuantifiedConditionsWithoutShortcut2KnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the 'QuantifiedConditionsWithoutShortcut2Generarator'
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT2] = KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT2

    seed: int = 42

    knowledge_share: float = 1.0

    number_of_bins: int = 5

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.qc_without_shortcut2 import QcWithoutShortcut2
        return QcWithoutShortcut2


class QuantifiedConditionsWithoutShortcut3KnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the 'QuantifiedConditionsWithoutShortcut3Generarator'
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT3] = KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITHOUT_SHORTCUT3

    seed: int = 42

    knowledge_share: float = 1.0

    number_of_bins: int = 5

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.qc_without_shortcut3 import QcWithoutShortcut3
        return QcWithoutShortcut3


class QuantifiedConditionsWithShortcutKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the 'QuantifiedConditionsWithShortcutKnowledgeGraphGenerator'
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT] = KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT

    seed: int = 42

    knowledge_share: float = 1.0

    number_of_bins: int = 5

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.qc_with_shortcut import QcWithShortcut
        return QcWithShortcut


class QuantifiedConditionsWithShortcut2KnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the 'QuantifiedConditionsWithShortcut2KnowledgeGraphGenerator'
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT2] = KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT2

    seed: int = 42

    knowledge_share: float = 1.0

    number_of_bins: int = 5

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.qc_with_shortcut2 import QcWithShortcut2
        return QcWithShortcut2


class QuantifiedConditionsWithShortcut3KnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the 'QuantifiedConditionsWithShortcut3KnowledgeGraphGenerator'
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT3] = KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_SHORTCUT3

    seed: int = 42

    knowledge_share: float = 1.0

    number_of_bins: int = 5

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.qc_with_shortcut3 import QcWithShortcut3
        return QcWithShortcut3


class QuantifiedConditionsWithLiteralKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the 'QuantifiedConditionsWithLiteralKnowledgeGraphGenerator'
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_LITERAL] = KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_WITH_LITERAL

    seed: int = 42

    knowledge_share: float = 1.0

    number_of_bins: int = 5

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.qc_with_literal import QcWithLiteral
        return QcWithLiteral


class QuantifiedConditionsW3KnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the 'QuantifiedConditionsW3KnowledgeGraphGenerator'
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_W3] = KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_W3

    seed: int = 42

    knowledge_share: float = 1.0

    number_of_bins: int = 5

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.qc_w3 import QcW3
        return QcW3


class QuantifiedConditionsW3WithLiteralKnowledgeGraphGeneratorConfig(AbstractKnowledgeGraphGeneratorConfig):
    """
    Configuration of the 'QuantifiedConditionsW3WithLiteralKnowledgeGraphGenerator'
    """
    type: Literal[KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_W3_WITH_LITERAL] = KnowledgeGraphGeneratorType.QUANTIFIED_CONDITIONS_W3_WITH_LITERAL

    seed: int = 42

    knowledge_share: float = 1.0

    number_of_bins: int = 5

    def get_generator_class(self):
        from data_provider.knowledge_graphs.generators.qc_w3 import QcW3
        return QcW3


KnowledgeGraphGeneratorConfig = Annotated[
    Union[
        KnowledgeGraphGeneratorType,
        BasicKnowledgeGraphGeneratorConfig,
        QuantifiedConditionsWithoutShortcutKnowledgeGraphGeneratorConfig,
        QuantifiedConditionsWithoutShortcut2KnowledgeGraphGeneratorConfig,
        QuantifiedConditionsWithoutShortcut3KnowledgeGraphGeneratorConfig,
        QuantifiedConditionsWithShortcutKnowledgeGraphGeneratorConfig,
        QuantifiedConditionsWithShortcut2KnowledgeGraphGeneratorConfig,
        QuantifiedConditionsWithShortcut3KnowledgeGraphGeneratorConfig,
        QuantifiedConditionsWithLiteralKnowledgeGraphGeneratorConfig,
        UnquantifiedKnowledgeGraphGeneratorConfig,
        QuantifiedParametersWithoutShortcutKnowledgeGraphGeneratorConfig,
        QuantifiedParametersWithoutShortcut2KnowledgeGraphGeneratorConfig,
        QuantifiedParametersWithoutShortcut3KnowledgeGraphGeneratorConfig,
        QuantifiedParametersWithShortcutKnowledgeGraphGeneratorConfig,
        QuantifiedParametersWithShortcut2KnowledgeGraphGeneratorConfig,
        QuantifiedParametersWithShortcut3KnowledgeGraphGeneratorConfig,
        QuantifiedParametersWithLiteralKnowledgeGraphGeneratorConfig,
        QuantifiedConditionsW3KnowledgeGraphGeneratorConfig,
        QuantifiedParametersW3KnowledgeGraphGeneratorConfig,
        QuantifiedParametersW3WithLiteralKnowledgeGraphGeneratorConfig,
        QuantifiedConditionsW3WithLiteralKnowledgeGraphGeneratorConfig,
    ],
    Field(discriminator='type')
]
"""Type alias that contains all available knowledge-graph-generator configuration classes."""


DEFAULT_KNOWLEDGE_GRAPH_GENERATOR_CONFIG: KnowledgeGraphGeneratorType = \
    KnowledgeGraphGeneratorType.BASIC
"""Default knowledge-graph-generator type that is used if no configuration is provided."""


def parse_knowledge_graph_generator_config(
    config: Optional[KnowledgeGraphGeneratorConfig]
) -> AbstractKnowledgeGraphGeneratorConfig:
    """
    Parses a knowledge-graph-generator configuration to its default format.

    If the configuration is None the default configuration is used.

    Parameters:
        config (KnowledgeGraphGeneratorConfig | None): config that should be parsed

    Returns:
        config in the default format (AbstractKnowledgeGraphGeneratorConfig)
    """
    if config is None:
        config = DEFAULT_KNOWLEDGE_GRAPH_GENERATOR_CONFIG
    if isinstance(config, KnowledgeGraphGeneratorType):
        return config.get_configuration()
    return config
