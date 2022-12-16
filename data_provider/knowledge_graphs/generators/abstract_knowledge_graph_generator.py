"""
This module provides the abstract class `KnowledgeGraphGenerator`.

To implement an own KnowledgeGraphGenerator follow the steps in
`data_provider.synthetic_data_generation.config.modules.knowledge_graph_generator_config.py`
"""

from abc import ABC, abstractmethod
from igraph import Graph

class KnowledgeGraphGenerator(ABC):
    """
    Abstract class that provides functionality to generate a knowledge graph representing the
    domain expert knowledge.
    """

    def __call__(self) -> Graph:
        """
        Generates a knowledge graph using the `generate_knowledge_graph` method.

        Returns:
            Generated knowledge graph
        """
        return self.generate_knowledge_graph()

    @abstractmethod
    def generate_knowledge_graph(self) -> Graph:
        """
        Generate a knowledge graph.

        Returns:
            Generated knowledge graph
        """
        
