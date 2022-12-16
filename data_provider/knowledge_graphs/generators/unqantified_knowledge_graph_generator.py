# pylint: disable=import-error, missing-function-docstring

from typing import List
from igraph import Graph
from data_provider.knowledge_graphs.generators.igraph_helper import add_vertex_to_graph
from numpy.random import default_rng as rng
from numpy.random import Generator

from data_provider.knowledge_graphs.config.knowledge_graph_generator_config \
    import UnquantifiedKnowledgeGraphGeneratorConfig
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.knowledge_graphs.generators.abstract_knowledge_graph_generator import KnowledgeGraphGenerator
from data_provider.synthetic_data_generation.types.generator_arguments \
    import KnowledgeGraphGeneratorArguments
from data_provider.synthetic_data_generation.types.pq_function import GeneratedPQFunctions
from data_provider.synthetic_data_generation.types.pq_tuple import PQTuple
from data_provider.knowledge_graphs.pq_relation import PQ_Relation


class UnquantifiedKnowledgeGraphGenerator(KnowledgeGraphGenerator):
    """
    Class that provides functionality to generate knowledge graphs.

    It randomly chooses pq-tuples from the expert knwoledge and adds them to the knowledge graph.
    The relations in this graph are unquantified.
    """

    _config: UnquantifiedKnowledgeGraphGeneratorConfig
    _expert_knowledge: List[PQTuple]
    _pq_functions: GeneratedPQFunctions
    _rng: Generator
    _sdg_config: SdgConfig

    def __init__(self, args: KnowledgeGraphGeneratorArguments) -> None:
        super().__init__()

        self._config = args.sdg_config.knowledge_graph_generator
        self._sdg_config = args.sdg_config
        self._relations = args.pq_relations
        self._rng = rng(self._config.seed)

    def generate_knowledge_graph(self) -> Graph:
        knowledge_graph = Graph(directed=True)

        num_included = int(len(self._relations) * self._config.knowledge_share)
        pq_tuples_included: List[PQ_Relation] = self._rng.choice(
            a=self._relations, size=num_included, replace=False
        )

        for pq_tuple in pq_tuples_included:
            added = add_vertex_to_graph(knowledge_graph, pq_tuple.quality)
            added["type"] = "qual_influence"

            added = add_vertex_to_graph(knowledge_graph, pq_tuple.parameter)
            added["type"] = "parameter"
            
            edge = knowledge_graph.add_edge(pq_tuple.quality, pq_tuple.parameter)
            edge['weight'] = pq_tuple.action
            

        return knowledge_graph