from abc import ABC
from typing import List
from igraph import Graph
from data_provider.knowledge_graphs.generators.igraph_helper import add_vertex_to_graph
from numpy.random import default_rng as rng
from numpy.random import Generator

from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_generation.types.generator_arguments import (
    KnowledgeGraphGeneratorArguments,
)

from data_provider.synthetic_data_generation.types.pq_function import (
    GeneratedPQFunctions,
)
from data_provider.synthetic_data_generation.types.pq_tuple import PQTuple
from data_provider.knowledge_graphs.pq_relation import PQ_Relation


class QC_super(ABC):
    """Superclass to unify all graph operations for the quantified_conditions like
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
        w3: bool = False,
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
        if w3 and any([flip_quantified_by, additional_flipped_quantified_by]):
            raise ValueError('flip_quantified_by and additional_flipped_quantified_by must be False if w3 is True!')

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

            if w3:
                self.add_w3_relation(pq_relation, knowledge_graph)
            else:
                self.add_relation(
                    pq_relation, knowledge_graph, with_shortcut,
                    flip_quantified_by, additional_flipped_quantified_by
                )

        return knowledge_graph

    def add_w3_relation(
        self,
        pq_relation: PQ_Relation,
        knowledge_graph: Graph
    ) -> None:
        """Adds a relation to the knowledge graph in the Style recommended by W3

        Args:
            pq_relation (PQ_Relation): the pq-relation to add
            knowledge_graph (iGraph.Graph): the Graph to add the relation to
        """
        parameter = pq_relation.parameter
        quality = pq_relation.quality
        bin_values = pq_relation.conclusion_quantifications

        pq_rel_name = parameter + "-" + quality + "-" + "relation"
        pq_rel_vertex = add_vertex_to_graph(knowledge_graph, pq_rel_name)
        pq_rel_vertex["type"] = "pq-relation"

        added = knowledge_graph.add_edge(quality, pq_rel_vertex)
        added["weight"] = "implies"
        added = knowledge_graph.add_edge(pq_rel_vertex, parameter)
        added["weight"] = "implied parameter"
        added["literal_included"] = "None"

        count = 0
        uid = 0
        for i, _bin in enumerate(pq_relation.condition_scopes):
            start_point = min(_bin)
            end_point = max(_bin)

            nu = f'{bin_values[i]}-for-{parameter}-{quality}|{uid}'
            uid += 1
            bin_start = f'{start_point}-for-{parameter}-{quality}|{uid}'
            uid += 1
            bin_end = f'{end_point}-for-{parameter}-{quality}|{uid}'
            uid += 1

            nu_vertex = add_vertex_to_graph(knowledge_graph, nu)
            # known vertices are appended to a list, get the end of the list to get the vertex object
            nu_vertex["type"] = "quantified_conclusion"
            nu_vertex["literal_value"] = str(bin_values[i])
            nu_vertex["corresponding_parameter"] = parameter
            nu_vertex["is_relative_value"] = pq_relation.action == PQ_Relation.Action.ADJUST
            lower_mu_vertex = add_vertex_to_graph(knowledge_graph, bin_start)
            lower_mu_vertex["type"] = "value"
            lower_mu_vertex["literal_value"] = str(start_point)
            upper_mu_vertex = add_vertex_to_graph(knowledge_graph, bin_end)
            upper_mu_vertex["type"] = "value"
            upper_mu_vertex["literal_value"] = str(end_point)

            mu_nu_name = f'my{count}-ny{count}-relation-for-{parameter}-{quality}'
            mu_nu_vertex = add_vertex_to_graph(knowledge_graph, mu_nu_name)
            mu_nu_vertex["type"] = "my-ny-relation"

            added = knowledge_graph.add_edge(pq_rel_vertex, mu_nu_vertex)
            added["weight"] = "quantified by"
            added["literal_included"] = "None"

            added = knowledge_graph.add_edge(mu_nu_vertex, lower_mu_vertex)
            added["weight"] = "starts at"
            added["literal_included"] = "To"

            added = knowledge_graph.add_edge(mu_nu_vertex, upper_mu_vertex)
            added["weight"] = "ends at"
            added["literal_included"] = "To"

            added = knowledge_graph.add_edge(mu_nu_vertex, nu_vertex)
            added["weight"] = f"{pq_relation.quantified_conclusion_prefix} quantified by"
            added["literal_included"] = "To"
            count += 1

    def add_relation(
        self,
        pq_relation: PQ_Relation,
        knowledge_graph: Graph,
        with_shortcut: bool,
        flip_quantified_by: bool,
        additional_flipped_quantified_by: bool
    ):
        """Adds a relation in the quantified_conditions style to a graph

        Args:
            pq_relation (PQ_Relation): the pq-relation to add
            knowledge_graph (iGraph.Graph): the knowledge graph who should be expanded
            with_shortcut (bool): wether a high level relation between p and q should be generated additionally
        """
        parameter = pq_relation.parameter
        quality = pq_relation.quality
        bin_values = pq_relation.conclusion_quantifications

        uid = 0
        for i, _bin in enumerate(pq_relation.condition_scopes):
            start_point = min(_bin)
            end_point = max(_bin)

            value = f'{_bin}-for-{parameter}-{quality}|{uid}'
            uid += 1
            bin_name = f'µ_{i}: {start_point}-{end_point}-for-{parameter}-{quality}'
            bin_start = f'{start_point}-for-{parameter}-{quality}|{uid}'
            uid += 1
            bin_end = f'{end_point}-for-{parameter}-{quality}|{uid}'
            uid += 1

            nu_vertex = add_vertex_to_graph(knowledge_graph, value)
            # new vertices are appended to a list, get the end of the list to get the vertex object
            nu_vertex["type"] = "quantified_conclusion"
            nu_vertex["literal_value"] = str(bin_values[i])
            nu_vertex["is_relative_value"] = pq_relation.action == PQ_Relation.Action.ADJUST

            mu_vertex = add_vertex_to_graph(knowledge_graph, bin_name)
            mu_vertex["type"] = "quantified_condition"

            lower_mu_ver = add_vertex_to_graph(knowledge_graph, bin_start)
            lower_mu_ver["type"] = "value"
            lower_mu_ver["literal_value"] = str(start_point)

            upper_mu_ver = add_vertex_to_graph(knowledge_graph, bin_end)
            upper_mu_ver["type"] = "value"
            upper_mu_ver["literal_value"] = str(end_point)

            # Add edges between the added vertices
            added = knowledge_graph.add_edge(mu_vertex, nu_vertex)
            added["weight"] = "implies"
            added["literal_included"] = "To"
            if flip_quantified_by or additional_flipped_quantified_by:
                added = knowledge_graph.add_edge(nu_vertex, parameter)
                added["weight"] = f"{pq_relation.quantified_conclusion_prefix} quantifies"
                added["literal_included"] = "From"
            if not flip_quantified_by or additional_flipped_quantified_by:
                added = knowledge_graph.add_edge(parameter, nu_vertex)
                added["weight"] = f"{pq_relation.quantified_conclusion_prefix} quantified by"
                added["literal_included"] = "To"
            added = knowledge_graph.add_edge(mu_vertex, lower_mu_ver)
            added["weight"] = "starts at"
            added["literal_included"] = "To"
            added = knowledge_graph.add_edge(mu_vertex, upper_mu_ver)
            added["weight"] = "ends at"
            added["literal_included"] = "To"
            added = knowledge_graph.add_edge(quality, mu_vertex)
            added["weight"] = "quantified by"
            added["literal_included"] = "None"
        if with_shortcut:
            added = knowledge_graph.add_edge(quality, parameter)
            added["weight"] = "implies"
            added["literal_included"] = "None"
