from abc import ABC
from typing import List
from igraph import Graph
from data_provider.knowledge_graphs.generators.igraph_helper import add_vertex_to_graph
from numpy.random import default_rng as rng
from numpy.random import Generator

from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_generation.types.generator_arguments import KnowledgeGraphGeneratorArguments

from data_provider.synthetic_data_generation.types.pq_function import GeneratedPQFunctions
from data_provider.synthetic_data_generation.types.pq_tuple import PQTuple
from data_provider.knowledge_graphs.pq_relation import PQ_Relation


class QP_super(ABC):
    """Superclass to unify all graph operations for the quantified_parameters like
    graphs.

    """
    _rng: Generator
    _expert_knowledge: List[PQTuple]
    _pq_functions: GeneratedPQFunctions
    _sdg_config: SdgConfig

    def __init__(self, args: KnowledgeGraphGeneratorArguments) -> None:
        self._config = args.sdg_config.knowledge_graph_generator
        self._sdg_config = args.sdg_config
        self._relations = args.pq_relations
        self._rng = rng(self._config.seed)

    def generalized_generate_knowledge_graph(
        self,
        with_shortcut: bool,
        flip_quantified_by: bool = False,
        additional_flipped_quantified_by: bool = False
    ) -> Graph:
        """
        Creates a knowledge graph

        Parameters:
            - `with_shortcut`: Whether to include a direct relation between quality and parameter
            - `flip_quantified_by`: Whether to flip the relation between the parameter and its
            quantification to be a 'quantifies' instead of a 'quantified by' relation
            - `additional_flipped_quantified_by`: Whether to include a 'quantifies' relation between
            the parameter and its quantification in addition to the 'quantified by' relation

        Returns:
            Created knowledge graph using the specified configurations
        """
        if flip_quantified_by and additional_flipped_quantified_by:
            raise ValueError('Only flip_quantified_by or additional_flipped_quantified_by can be set to True, not both!')

        knowledge_graph = Graph(directed=True)

        num_included = int(len(self._relations) * self._config.knowledge_share)
        pq_relations_included: List[PQ_Relation] = self._rng.choice(
            a=self._relations, size=num_included, replace=False
        )

        for pq_relation in pq_relations_included:
            added = add_vertex_to_graph(knowledge_graph, pq_relation.quality)
            added["type"] = "qual_influence"

            added = add_vertex_to_graph(knowledge_graph, pq_relation.parameter)
            added["type"] = "parameter"

            parameter = pq_relation.parameter
            quality = pq_relation.quality

            edge_weight = pq_relation.conclusion_quantification_mean

            quant = add_vertex_to_graph(knowledge_graph, edge_weight)
            quant['type'] = "value"
            quant['literal_value'] = edge_weight
            quant['corresponding_parameter'] = parameter
            quant["is_relative_value"] = pq_relation.action == PQ_Relation.Action.ADJUST

            added = knowledge_graph.add_edge(quality, quant)
            added["weight"] = "implies"
            added["literal_included"] = "To"

            if flip_quantified_by or additional_flipped_quantified_by:
                added = knowledge_graph.add_edge(quant, parameter)
                added['weight'] = f"{pq_relation.quantified_conclusion_prefix} quantifies"
                added["literal_included"] = "From"
            if not flip_quantified_by or additional_flipped_quantified_by:
                added = knowledge_graph.add_edge(parameter, quant)
                added['weight'] = f"{pq_relation.quantified_conclusion_prefix} quantified by"
                added["literal_included"] = "To"

            if with_shortcut:
                added = knowledge_graph.add_edge(quality, parameter)
                added['weight'] = "implies"
                added['literal_included'] = 'None'

        return knowledge_graph
