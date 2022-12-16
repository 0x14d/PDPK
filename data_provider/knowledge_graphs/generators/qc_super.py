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
        self, with_shortcut: bool, with_nary=False
    ) -> Graph:
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

            pair = (pq_tuple.parameter, pq_tuple.quality)

            if with_nary:
                self.add_w3_relation(
                    pair, knowledge_graph, pq_tuple.condition_scopes, pq_tuple.condition_values
                )
            else:
                self.add_relation(
                    pair,
                    knowledge_graph,
                    pq_tuple.action,
                    pq_tuple.condition_scopes,
                    pq_tuple.condition_values,
                    with_shortcut,
                )
        return knowledge_graph

    def add_w3_relation(self, pq_tuple, knowledge_graph, bins, bin_values):
        """Adds a relation to the knowledge graph in the Style recommended by W3

        Args:
            pq_tuple (tuple(parameter: str,quality : str)): the pq-relation to add
            knowledge_graph (iGraph.Graph): the Graph to add the relation to
        """
        parameter = pq_tuple[0]
        quality = pq_tuple[1]

        nary_name = parameter + "-" + quality + "-" + "relation"
        nary = add_vertex_to_graph(knowledge_graph, nary_name)
        pq_rel = knowledge_graph.vs.select()[-1]
        pq_rel["type"] = "pq-relation"

        added = knowledge_graph.add_edge(quality, nary)
        added["weight"] = "implies"
        added = knowledge_graph.add_edge(nary, parameter)
        added["weight"] = "implied parameter"
        added["literal_included"] = "None"

        count = 0
        uid = 0
        for i in range(len(bins)):
            # Check which bin edge is lower and then set the start point to the
            # lower boundary
            if bins[i][0] < bins[i][1]:
                start_point = bins[i][0]
                end_point = bins[i][1]
            elif bins[i][0] > bins[i][1]:
                start_point = bins[i][1]
                end_point = bins[i][0]

            value = (
                str(bin_values[i])
                + "-for-"
                + parameter
                + "-"
                + quality
                + "|"
                + str(uid)
            )
            uid += 1
            bin = (
                ("µ_" + str(i) + ": " + str(start_point) + "-" + str(end_point))
                + "-for-"
                + parameter
                + "-"
                + quality
            )
            bin_start = (
                str(start_point) + "-for-" + parameter + "-" + quality + "|" + str(uid)
            )
            uid += 1
            bin_end = (
                str(end_point) + "-for-" + parameter + "-" + quality + "|" + str(uid)
            )
            uid += 1

            value_vertex = add_vertex_to_graph(knowledge_graph, value)
            # Knew vertices are appended to a list, get the end of the list to get the vertex object
            value_vertex["type"] = "quantified_conclusion"
            value_vertex["literal_value"] = str(bin_values[i])
            value_vertex["corresponding_parameter"] = parameter
            bin_vertex = add_vertex_to_graph(knowledge_graph, bin)
            bin_vertex["type"] = "quantified_condition"
            start_vertex = add_vertex_to_graph(knowledge_graph, bin_start)
            start_vertex["type"] = "value"
            start_vertex["literal_value"] = str(start_point)
            end_vertex = add_vertex_to_graph(knowledge_graph, bin_end)
            end_vertex["type"] = "value"
            end_vertex["literal_value"] = str(end_point)

            rel_name = (
                "my"
                + str(count)
                + "-"
                + "ny"
                + str(count)
                + "-relation"
                + "-for-"
                + parameter
                + "-"
                + quality
            )
            myny_rel = add_vertex_to_graph(knowledge_graph, rel_name)
            rel_vertex = knowledge_graph.vs.select()[-1]
            rel_vertex["type"] = "my-ny-relation"

            added = knowledge_graph.add_edge(pq_rel, myny_rel)
            added["weight"] = "quantified by"
            added["literal_included"] = "None"

            added = knowledge_graph.add_edge(myny_rel, start_vertex)
            added["weight"] = "starts at"
            added["literal_included"] = "To"

            added = knowledge_graph.add_edge(myny_rel, end_vertex)
            added["weight"] = "ends at"
            added["literal_included"] = "To"

            added = knowledge_graph.add_edge(myny_rel, value_vertex)
            added["weight"] = "quantified by"
            added["literal_included"] = "To"
            count += 1

    def add_relation(self, pq_tuple, knowledge_graph, action, bins, bin_values, with_shortcut):
        """Adds a relation in the quantified_conditions style to a graph

        Args:
            pq_tuple (tuple(parameter : str, qualtiy : str)): the pq-relation to add
            knowledge_graph (iGraph.Graph): the knowledge graph who should be expanded
            with_shortcut (bool): wether a high level relation between p and q should be generated additionally
        """
        parameter = pq_tuple[0]
        quality = pq_tuple[1]

        uid = 0
        for i in range(len(bins)):
            # Check which bin edge is lower and then set the start point to the
            # lower boundary
            if bins[i][0] < bins[i][1]:
                start_point = bins[i][0]
                end_point = bins[i][1]
            elif bins[i][0] > bins[i][1]:
                start_point = bins[i][1]
                end_point = bins[i][0]
                
            value = str(bins[i]) + "-for-" + parameter + "-" + quality + "|" + str(uid)
            uid += 1
            bin = (
                ("µ_" + str(i) + ": " + str(start_point) + "-" + str(end_point))
                + "-for-"
                + parameter
                + "-"
                + quality
            )
            bin_start = (
                str(start_point) + "-for-" + parameter + "-" + quality + "|" + str(uid)
            )
            uid += 1
            bin_end = (
                str(end_point) + "-for-" + parameter + "-" + quality + "|" + str(uid)
            )
            uid += 1

            value_vertex = add_vertex_to_graph(knowledge_graph, value)
            # new vertices are appended to a list, get the end of the list to get the vertex object
            value_vertex["type"] = "quantified_conclusion"
            value_vertex["literal_value"] = str(bin_values[i])

            bin_vertex = add_vertex_to_graph(knowledge_graph, bin)
            bin_vertex["type"] = "quantified_condition"

            start_vertex = add_vertex_to_graph(knowledge_graph, bin_start)
            start_vertex["type"] = "value"
            start_vertex["literal_value"] = str(start_point)

            end_vertex = add_vertex_to_graph(knowledge_graph, bin_end)
            end_vertex["type"] = "value"
            end_vertex["literal_value"] = str(end_point)

            # Add edges between the added vertices
            added = knowledge_graph.add_edge(bin_vertex, value_vertex)
            added["weight"] = "implies"
            added["literal_included"] = "To"
            added = knowledge_graph.add_edge(parameter, value_vertex)
            added["weight"] = "quantified by"
            added["literal_included"] = "To"
            added = knowledge_graph.add_edge(bin_vertex, start_vertex)
            added["weight"] = "starts at"
            added["literal_included"] = "To"
            added = knowledge_graph.add_edge(bin_vertex, end_vertex)
            added["weight"] = "ends at"
            added["literal_included"] = "To"
            added = knowledge_graph.add_edge(quality, bin_vertex)
            added["weight"] = "quantified by"
            added["literal_included"] = "None"
        if with_shortcut:
            added = knowledge_graph.add_edge(quality, parameter)
            added["weight"] = action
            added["literal_included"] = "None"
