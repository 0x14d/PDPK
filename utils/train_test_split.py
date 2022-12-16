"""
Provides functionality to split a knowledge graph into a train and test split.
"""

# pylint: disable=import-error

from typing import Optional, Tuple

import pandas as pd
from pykeen.triples import TriplesFactory, TriplesNumericLiteralsFactory, CoreTriplesFactory
from sklearn.model_selection import train_test_split

from utils.relation_list import create_relation_list
from data_provider.knowledge_graphs.config.knowledge_graph_generator_config \
    import KnowledgeGraphGeneratorType


def kg_train_test_split(
    kg_type: KnowledgeGraphGeneratorType,
    edges: pd.DataFrame,
    metadata: pd.DataFrame,
    seed,
    test_split: float,
    use_literals: bool,
) -> Tuple[CoreTriplesFactory, Optional[CoreTriplesFactory]]:
    """
    Splits a kg into a train and test split.
    If possible it uses the pykeen `CoreTriplesFactory` to split the kg.
    Otherwise a basic sklearn `train_test_split` is applied.

    Parameters:
        - kg_type (KnowledgeGraphGeneratorType): representation of the kg
        - edges (pd.DataFrame): edges of the kg
        - metadata (pd.DataFrame): verticies of the kg
        - seed (any): seed / random state for the rng
        - test_split (float): proportion of the kg that is used as test split
        - use_literals (bool): defines if the train-test-split should include literals

    Returns:
        train split, test split (if `test_split` != 0.0 else None)
    """
    relations, literals = create_relation_list(kg_type, edges, metadata)
    mapping = {}
    for index, _ in metadata.iterrows():
        # Do not add literals to the mapping
        if kg_type == 'quantified_parameters_with_literal' and metadata.loc[index]['type'] is None:
            continue
        mapping[str(metadata.loc[index]['name'])] = index

    args = {
        'relations': relations,
        'mapping': mapping,
        'seed': seed,
        'test_split': test_split
    }
    if use_literals:
        return _kg_train_test_split_literal(**args, literals=literals)
    else:
        return _kg_train_test_split(**args)

def _kg_train_test_split(
    relations,
    mapping,
    seed,
    test_split: float
) -> Tuple[TriplesFactory, Optional[TriplesFactory]]:
    """Creates the train-test-split without literals"""
    factory = TriplesFactory.from_labeled_triples(triples=relations, entity_to_id=mapping)

    if test_split == 0.0:
        return factory, None

    try:
        # For the Test-Train split the default pykeen-method is used.
        # For very small graphs this method will fail, since pykeen tries
        # to find full coverage of every type on both the test and train
        # data. When this error raises, the scikit test-train split is
        # used instead.
        return factory.split(ratios= 1 - test_split, random_state=seed)
    except ValueError:
        train_raw, test_raw = train_test_split(relations, test_size=test_split, random_state=seed)
        train_data = TriplesFactory.from_labeled_triples(triples=train_raw, entity_to_id=mapping)
        test_data = TriplesFactory.from_labeled_triples(triples=test_raw, entity_to_id=mapping)
        return train_data, test_data

def _kg_train_test_split_literal(
    relations,
    mapping,
    literals,
    seed,
    test_split: float
) -> Tuple[TriplesNumericLiteralsFactory, Optional[TriplesNumericLiteralsFactory]]:
    """Creates the train-test-split with literals"""
    factory = TriplesNumericLiteralsFactory.from_labeled_triples(
        triples=relations,
        numeric_triples=literals,
        entity_to_id=mapping
    )

    if test_split == 0.0:
        return factory, None

    try:
        # For the Test-Train split the default pykeen-method is used.
        # For very small graphs this method will fail, since pykeen tries
        # to find full coverage of every type on both the test and train
        # data. When this error raises, the scikit test-train split is
        # used instead.
        return factory.split(ratios= 1 - test_split, random_state=seed)
    except ValueError:
        train_raw, test_raw = train_test_split(relations, test_size=test_split, random_state=seed)
        train_data = TriplesNumericLiteralsFactory.from_labeled_triples(
            triples=train_raw,
            numeric_triples=literals,
            entity_to_id=mapping
        )
        test_data = TriplesNumericLiteralsFactory.from_labeled_triples(
            triples=test_raw,
            numeric_triples=literals,
            entity_to_id=mapping
        )
        return train_data, test_data
