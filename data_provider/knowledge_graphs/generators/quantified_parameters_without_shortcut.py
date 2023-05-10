# pylint: disable=import-error, missing-function-docstring

from igraph import Graph

from data_provider.knowledge_graphs.generators.abstract_knowledge_graph_generator import KnowledgeGraphGenerator
from data_provider.synthetic_data_generation.types.generator_arguments \
    import KnowledgeGraphGeneratorArguments
from data_provider.knowledge_graphs.generators.qp_super import QP_super


class QuantifiedParametersWithoutShortcut(KnowledgeGraphGenerator, QP_super):
    """
    Class that provides functionality to generate knowledge graphs that display
    a quantified parameters like relation missing a high level pq-relation
    """

    def __init__(self, args: KnowledgeGraphGeneratorArguments) -> None:
        KnowledgeGraphGenerator.__init__(self)
        QP_super.__init__(self,args)

    def generate_knowledge_graph(self) -> Graph:
        return QP_super.generalized_generate_knowledge_graph(self, with_shortcut=False)