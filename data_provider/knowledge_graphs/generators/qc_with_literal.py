from igraph import Graph

from data_provider.knowledge_graphs.generators.abstract_knowledge_graph_generator import \
    KnowledgeGraphGenerator
from data_provider.synthetic_data_generation.types.generator_arguments import KnowledgeGraphGeneratorArguments
from data_provider.knowledge_graphs.generators.qc_super import QC_super


class QcWithLiteral(KnowledgeGraphGenerator, QC_super):
    """
    Class that provides functionality to generate knowledge graphs that display
    a quantified conditions like relation including literal entities
    """


    def __init__(self, args: KnowledgeGraphGeneratorArguments) -> None:
        KnowledgeGraphGenerator.__init__(self)
        QC_super.__init__(self,args)

    def generate_knowledge_graph(self) -> Graph:
        return QC_super.generalized_generate_knowledge_graph(self,with_shortcut=True)
