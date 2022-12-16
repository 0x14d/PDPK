"""Provides functionality to create a relation list for a knowledge graph"""

# pylint: disable=import-error

from typing import Tuple

import numpy as np
import pandas as pd

from data_provider.knowledge_graphs.config.knowledge_graph_generator_config \
    import KnowledgeGraphGeneratorType


def create_relation_list(
    kg_type: KnowledgeGraphGeneratorType,
    edges: pd.DataFrame,
    metadata: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a list of h,r,t relations for the embedding libraries to use.
    If a representation contains a 'with_literal' relations containing literals
    will be put into a seperate literal-list, for special embedding methods.

    Parameters:
        - kg_type (KnowledgeGraphGeneratorType): type of the kg
        - edges (pd.DataFrame): dataframe containing all edges of the kg
        - metadata (pd.DataFrame): dataframe containing all verticies of the kg

    Returns:
        List of relations and literals
    """

    literal_relations = []
    entity_relations = []

    if 's_with_literal' in kg_type:
        for _, row in edges.iterrows():
            try:
                row.loc['literal_included']
            except:
                pass
            source_int = row.loc['from']
            to_int = row.loc['to']
            rel_float = row.loc['rel']
            source_name = metadata.loc[source_int]['name']
            to_name = metadata.loc[to_int]['name']
            if row.loc['literal_included'] == 'From':
                raise ValueError(
                    "Relations should not be formed this way in our current specification " +
                    "since pykeen requires all literals to be in the form NODE->RELATION->LITERAL"
                )
            elif row.loc['literal_included'] == 'To':
                literal_value = metadata.loc[to_int]['literal_value']
                literal_relations.append(
                    [source_name, str(rel_float), literal_value])
            elif row.loc['literal_included'] == "None":
                entity_relations.append(
                    [source_name, str(rel_float), to_name])

    elif "w3_with_literal" in kg_type:
        for _, row in edges.iterrows():
            source_int = row.loc['from']
            to_int = row.loc['to']
            rel_float = row.loc['rel']
            source_name = metadata.loc[source_int]['name']
            to_name = metadata.loc[to_int]['name']
            if row.loc['literal_included'] == 'From':
                literal_value = metadata.loc[source_int]['literal_value']
                literal_relations.append(
                    [to_name, str(rel_float), literal_value])
            elif row.loc['literal_included'] == 'To':
                literal_value = metadata.loc[to_int]['literal_value']
                literal_relations.append(
                    [source_name, str(rel_float), literal_value])
            elif row.loc['literal_included'] == "None":
                entity_relations.append(
                    [source_name, str(rel_float), to_name])

    else:
        for _, row in edges.iterrows():
            source_int = row.loc['from']
            to_int = row.loc['to']
            rel_float = row.loc['rel']
            source_name = metadata.loc[source_int]['name']
            to_name = metadata.loc[to_int]['name']
            entity_relations.append([source_name, str(rel_float), to_name])

    np_rel = np.array(entity_relations)
    np_lit = np.array(literal_relations)

    if 'with_literal' not in kg_type:
        np_lit = np.array([[0, 0, 0]])
    return (np_rel, np_lit)
