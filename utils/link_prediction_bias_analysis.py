"""
Provides the class `LinkPredictionBiasAnalysis` to analyze the
bias of the link prediction train-test-split based on Rossi et al
(https://alammehwish.github.io/dl4kg2021/papers/knowledge_graph_embeddings_or_.pdf).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import igraph
import pandas as pd

from data_provider.knowledge_graphs.config.knowledge_graph_generator_config \
    import KnowledgeGraphGeneratorType
from utils.train_test_split import kg_train_test_split

@dataclass
class TripleBiasResult:
    """Bias results for a single kg triple"""

    type1_head_bias: float
    """Type 1 head bias value"""

    type2_head_bias: float
    """Type 2 head bias value"""

    type3_head_bias: float
    """Type 3 head bias value"""

    type1_tail_bias: float
    """Type 1 tail bias value"""

    type2_tail_bias: float
    """Type 2 tail bias value"""

    type3_tail_bias: float
    """Type 3 tail bias value"""

    count: int = field(init=False, default=1)
    """Count how ofter this triple occures in the data (same tuples have the same results)"""

@dataclass
class LinkPredictionBiasAnalysis:
    """
    Class that provides functionality to calculate the bias of the link prediction train-test-split
    based on

    [1] Rossi, A., D. Firmani, and P. Merialdo. "Knowledge graph embeddings or bias graph embeddings?
    a study of bias in link prediction models." (2021).
    (https://alammehwish.github.io/dl4kg2021/papers/knowledge_graph_embeddings_or_.pdf).

    This implementation is based on https://github.com/merialdo/research.lpbias.
    """

    train_data: List[Tuple[str, str, str]]
    """List of all train data tuples"""

    test_data: List[Tuple[str, str, str]]
    """List of all test data tuples"""

    type1_threshold: float = 0.75
    """Threshold for the type 1 bias"""

    type2_threshold: float = 0.5
    """Threshold for the type 2 bias"""

    type3_threshold: float = 0.5
    """Threshold for the type 3 bias"""

    results: Dict[Tuple[str, str, str], TripleBiasResult] = \
        field(init=False, default_factory=lambda: {})
    """Results of the bias analysis"""

    def __post_init__(self) -> None:
        # Calculate biases
        for triple in self.test_data:
            if triple in self.results:
                self.results[triple].count += 1
                continue

            type1_head_bias, type1_tail_bias = self._calculate_type_1_bias(*triple)
            type2_head_bias, type2_tail_bias= self._calculate_type_2_bias(*triple)
            type3_head_bias, type3_tail_bias = self._calculate_type_3_bias(*triple)
            self.results[triple] = TripleBiasResult(
                type1_head_bias=type1_head_bias,
                type2_head_bias=type2_head_bias,
                type3_head_bias=type3_head_bias,
                type1_tail_bias=type1_tail_bias,
                type2_tail_bias=type2_tail_bias,
                type3_tail_bias=type3_tail_bias
            )

    def _get_triples(
        self,
        head: Optional[str]=None,
        relation: Optional[str]=None,
        tail: Optional[str]=None,
        use_train: bool=True,
        use_test: bool = False
    ) -> List[Tuple[str, str, str]]:
        """
        Returns all triples that match the specified filters

        Parmeters:
            - head (str, optional): If provided only return triples with matching head
            - realtion (str, optional): If provided only return triples with matching relation
            - tail (str, optional): If provided only return triples with matching tail
            - use_train(bool, optional, default: True):
                If True include triples from the train set, else not
            - use_test(bool, optional, default: False):
                If True include triples from the test set, else not

        Returns:
            List of all triples matching the specified criteria
        """
        triples = (self.train_data if use_train else []) + (self.test_data if use_test else [])
        return [
            (h, r, t) for h, r, t in triples
            if head in [h, None] and relation in [r, None] and tail in [t, None]
        ]

    def _calculate_type_1_bias(self, head: str, relation: str, tail: str) -> Tuple[float, float]:
        """
        Calculates the type 1 bias for the specified triple (head, relation, tail).
        It is defined as follows:

        "A tail prediction ⟨ℎ, 𝑟, 𝑡⟩ is prone to Type 1 Bias if the training facts mentioning
        𝑟 tend to always feature 𝑡 as tail.
        For example, the tail prediction ⟨𝐵𝑎𝑟𝑎𝑐𝑘_𝑂𝑏𝑎𝑚𝑎, 𝑔𝑒𝑛𝑑𝑒𝑟, 𝑚𝑎𝑙𝑒⟩ is prone to this type
        of bias if the vast majority of gendered entities in the training set are males:
        this artificially favours the prediction of male genders.
        In practice, we verify if the fraction between the number of training facts featuring
        both 𝑟 and 𝑡 and the number of training facts featuring 𝑟 exceeds a threshold 𝜏1." [1]

        Returns:
            head bias value, tail bias value
        """
        count_relation = float(len(self._get_triples(relation=relation)))
        if count_relation == 0:
            return math.nan, math.nan
        head_bias = len(self._get_triples(relation=relation, head=head)) / count_relation
        tail_bias = len(self._get_triples(relation=relation, tail=tail)) / count_relation
        return head_bias, tail_bias

    def _calculate_type_2_bias(self, head: str, relation: str, tail: str) -> Tuple[float, float]:
        """
        Calculates the type 2 bias for the specified triple (head, relation, tail).
        It is defined as follows:

        "A tail prediction ⟨ℎ, 𝑟, 𝑡⟩ in which 𝑟 is a one-to-many or a many-to-many relation
        is prone to Type 2 Bias if, whenever an entity 𝑒 is seen as head for relation 𝑟,
        fact ⟨𝑒, 𝑟, 𝑡⟩ also exists in 𝒢𝑡𝑟𝑎𝑖𝑛.
        Type 2 Bias affects relations that have a "default" correct answer.
        Differently from Type 1, facts mentioning 𝑟 may feature a variety of tails different from 𝑡;
        however, for each entity 𝑒 seen as head these facts,
        𝑡 tends to always be among the correct tails too.
        This makes ⟨𝑒, 𝑟, 𝑡⟩ artificially easier to predict.
        For instance, the tail prediction ⟨𝐶𝑟𝑖𝑠𝑡𝑖𝑎𝑛𝑜_𝑅𝑜𝑛𝑎𝑙𝑑𝑜, 𝑙𝑎𝑛𝑔𝑢𝑎𝑔𝑒, 𝐸𝑛𝑔𝑙𝑖𝑠ℎ⟩ is prone to
        Type 2 Bias if most people, in addition to other languages, also speak English.
        In practice, we verify if the fraction of entities 𝑒 seen as heads for relation 𝑟 and
        that also display a fact ⟨𝑒, 𝑟, 𝑡⟩ exceeds a threshold 𝜏2." [1]

        Returns:
            head bias value, tail bias value
        """

        def get_relation_multiplicities(relation: str) -> Tuple[int, int]:
            """
            Counts how ofter the most occuring head and tail occur for the specified relation.

            Returns:
                occurrences of the most occuring head, occurrences of the most occuring tail
            """
            triples = self._get_triples(relation=relation, use_train=True, use_test=True)
            heads = {h for h, _, _ in triples}
            tails = {t for _, _, t in triples}

            m_tail = max([
               len({
                    t for _, _, t in
                    self._get_triples(head=h, relation=relation, use_train=True, use_test=True)
                }) for h in heads
            ])
            m_head = max([
               len({
                    h for h, _, _ in
                    self._get_triples(tail=t, relation=relation, use_train=True, use_test=True)
                }) for t in tails
            ])

            return m_head, m_tail

        m_head, m_tail = get_relation_multiplicities(relation)

        head_bias = math.nan
        if m_tail > 1:
            tails_with_relation = {t for _, _, t in self._get_triples(relation=relation)}
            tails_with_relation_and_head = {
                t for _, _, t in self._get_triples(relation=relation, head=head)
            }
            if len(tails_with_relation) != 0:
                head_bias = float(len(tails_with_relation_and_head)) / len(tails_with_relation)

        tail_bias = math.nan
        if m_head > 1:
            heads_with_relation = {h for h, _, _ in self._get_triples(relation=relation)}
            heads_with_relation_and_tails = {
                h for h, _, _ in self._get_triples(relation=relation, tail=tail)
            }
            if len(heads_with_relation) != 0:
                tail_bias = float(len(heads_with_relation_and_tails)) / len(heads_with_relation)

        return head_bias, tail_bias

    def _calculate_type_3_bias(self, head: str, relation: str, tail: str) -> Tuple[float, float]:
        """
        Calculates the type 1 bias for the specified triple (head, relation, tail).
        It is defined as follows:

        "A tail prediction ⟨ℎ, 𝑟, 𝑡⟩ is prone to Type 3 Bias if a relation 𝑠 exists such that:
            - (i) whenever 𝑠 links two entities, 𝑟 links them as well; and
            - (ii) the fact ⟨ℎ, 𝑠, 𝑡⟩ is present in the training set.

        For example, in the FB15k dataset the producer of a TV program is almost always its creator
        too; this may lead to assume that creating a program implies being its producer.
        In practice, to verify if 𝑠 and 𝑟 share this correlation we check if the fraction
        of 𝑠 mentions in which 𝑠 also co-occurs with 𝑟 is greater than a threshold 𝜏3." [1]

        Returns:
            head bias value, tail bias value
        """

        def get_dominating_relations(relation: str) -> Set[str]:
            """
            Finds the relations that share the matching heads and tails with the specified relation.
            The minimal fraction of matching (head, tail) pairs is defined by the type3 threshold.

            Retuns:
                Set of all intersecting relations
            """
            relation_heads_and_tails = {(h, t) for h, _, t in self._get_triples(relation=relation)}
            other_relations = {r for _, r, _ in self.train_data + self.test_data if r != relation}
            dominating_relations = set()
            for other_relation in other_relations:
                other_relation_heads_and_tails = {
                    (h, t) for h, _, t
                    in self._get_triples(relation=other_relation)
                }
                matches = [
                    h_t for h_t in relation_heads_and_tails
                    if h_t in other_relation_heads_and_tails
                ]

                if len(other_relation_heads_and_tails) != 0 and \
                   float(len(matches)) / len(other_relation_heads_and_tails) > self.type3_threshold:
                    dominating_relations.add(other_relation)

            return dominating_relations

        dominating_relations = get_dominating_relations(relation)
        biased_relations = [
            dominating_relation for dominating_relation in dominating_relations
            if (head, dominating_relation, tail) in self.train_data
        ]
        bias = 0 if len(biased_relations) == 0 else 1
        return bias, bias

    def apply_thresholds(
        self,
        bias_types: List[int],
        head_tail: str
    ) -> Dict[Tuple[str, str, str], TripleBiasResult]:
        """
        Filters out all the result that reach the thresholds for the specified bias types.

        Parameters:
            - bias_types (List[int]): Bias types for which the threshold should be applied
            - head_tail (str): Defines wheter the 'head' or 'tail' results should be filtered

        Returns:
            All results that don't reached the thresholds for the specified bias types
        """
        def threshold_filter(result: TripleBiasResult) -> bool:
            """
            Returns if the specified result should be kept or filtered out

            Returns:
                False if the result should be filtered out, else True
            """
            for bias_type in bias_types:
                value = getattr(result, f'type{bias_type}_{head_tail}_bias')
                if math.isnan(value):
                    return True
                threshold = getattr(self, f'type{bias_type}_threshold') if bias_type != 3 else 1.0
                if value >= threshold:
                    return False
            return True

        return {
            k: v for k, v in self.results.items() if threshold_filter(v)
        }

def link_prediction_bias_analysis(
    graph: igraph.Graph,
    kg_type: KnowledgeGraphGeneratorType,
    use_literals: bool = False,
    test_split: float = 0.2,
    seed: Any = 1111,
    type1_threshold: float = 0.75,
    type2_threshold: float = 0.5,
    type3_threshold: float = 0.5
) -> LinkPredictionBiasAnalysis:
    """
    Creates an `LinkPredictionBiasAnalysis` object for the specified kg.

    Parameters:
        - graph (igraph.Graph): graph that should be analyzed
        - kg_type(KnowledgeGraphGeneratorType): representation of the kg
        - use_literals (bool): include literals in data?
        - test_split (float): proportion of the data that is included in the test split
        - seed (Any): random seed for the train-tests-split
        - type1_threshold (float): threshold used for the type 1 bias
        - type2_threshold (float): threshold used for the type 2 bias
        - type3_threshold (float): threshold used for the type 3 bias
    
    Returns:
        `LinkPredictionBiasAnalysis` object containing the analysis results
    """
    edges = graph.get_edge_dataframe().rename(
        columns={'source': 'from', 'target': 'to', 'weight': 'rel'}
    )
    train_data, test_data = kg_train_test_split(
        kg_type=kg_type,
        edges=edges,
        metadata=graph.get_vertex_dataframe(),
        test_split=test_split,
        seed=seed,
        use_literals=use_literals
    )
    train_data = [tuple(r) for r in train_data.triples.tolist()]
    test_data = [tuple(r) for r in test_data.triples.tolist()]
    analysis = LinkPredictionBiasAnalysis(
        train_data=train_data,
        test_data=test_data,
        type1_threshold=type1_threshold,
        type2_threshold=type2_threshold,
        type3_threshold=type3_threshold
    )

    return analysis

def create_dataframe(results: Dict[str, LinkPredictionBiasAnalysis]) -> pd.DataFrame:
    """
    Creates a dataframe containing detailed analysis of the link prediction bias results.

    Parameters:
        - results: Dictonary that maps the row indexes to their bias analysis results
    
    Returns:
        Analysis dataframe
    """
    def count_string(res: Dict[Tuple[str, str, str], TripleBiasResult], base_count: int) -> str:
        """
        Returns a formated string that includes the number of triples that didn't
        get filtered and the percentage of how many triples got filtered out.
        """
        new_count = sum(r.count for r in res.values())
        change = round(1 - new_count / float(base_count), 2) * -1
        return f'{new_count} ({change}%)'

    data = [
        [
            len(res.test_data),
            len(res.test_data),
            count_string(res.apply_thresholds([1], 'head'), len(res.test_data)),
            count_string(res.apply_thresholds([1], 'tail'), len(res.test_data)),
            count_string(res.apply_thresholds([2], 'head'), len(res.test_data)),
            count_string(res.apply_thresholds([2], 'tail'), len(res.test_data)),
            count_string(res.apply_thresholds([3], 'head'), len(res.test_data)),
            count_string(res.apply_thresholds([3], 'tail'), len(res.test_data)),
            count_string(res.apply_thresholds([1, 2, 3], 'head'), len(res.test_data)),
            count_string(res.apply_thresholds([1, 2, 3], 'tail'), len(res.test_data)),
        ]
        for res in results.values()
    ]
    mdix = pd.MultiIndex.from_product([
            ['Test Predictions'], ['Test size', 'w/o B1', 'w/o B2', 'w/o B3', 'w/o B*'], ['Head', 'Tail']
    ])
    return pd.DataFrame(data, columns=mdix, index=results.keys())


if __name__ == '__main__':
    import argparse
    import os
    from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
    from data_provider.synthetic_data_generation.synthetic_data_generator import SyntheticDataGenerator
    from data_provider.knowledge_graphs.config.knowledge_graph_generator_config \
        import parse_knowledge_graph_generator_config
    
    parser = argparse.ArgumentParser(
        description="run link prediction bias analysis"
    )
    parser.add_argument(
        '--sdg-config',
        help='path to the sdg config file',
        default='configs/default_config_sdg.json',
        type=str
    )
    parser.add_argument(
        '--use-literals',
        help='add the literals to the data (or not)',
        default=False,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--test-split',
        help='proportion of the data that is included in the test split',
        default=0.2,
        type=float
    )
    parser.add_argument(
        '--seed',
        help='seed for the train-test-split',
        default=1111,
    )
    parser.add_argument(
        '--type1-threshold',
        help='threshold for the type 1 bias',
        default=0.75,
        type=float
    )
    parser.add_argument(
        '--type2-threshold',
        help='threshold for the type 2 bias',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--type3-threshold',
        help='threshold for the type 3 bias',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--all-representations',
        help='evaluate all kg representations (or not)',
        default=False,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--output-dir',
        help='path where the output file gets saved to',
        default='outputs',
        type=str
    )
    args = parser.parse_args()

    sdg_config = SdgConfig.create_config(args.sdg_config)
    if args.all_representations:
        kg_configs = [parse_knowledge_graph_generator_config(kg) for kg in KnowledgeGraphGeneratorType]
    else:
        kg_configs = [sdg_config.knowledge_graph_generator]
    
    results: Dict[KnowledgeGraphGeneratorType, LinkPredictionBiasAnalysis] = {}
    for kg_config in kg_configs:
        sdg_config.knowledge_graph_generator = kg_config
        sdg = SyntheticDataGenerator(sdg_config)
        results[kg_config.type.value] = link_prediction_bias_analysis(
            graph=sdg.knowledge_graph,
            kg_type=kg_config.type,
            use_literals=args.use_literals,
            test_split=args.test_split,
            seed=args.seed,
            type1_threshold=args.type1_threshold,
            type2_threshold=args.type2_threshold,
            type3_threshold=args.type3_threshold,
        )
    
    dataframe = create_dataframe(results)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    file = os.path.join(args.output_dir, 'bias_analysis.xlsx')
    dataframe.to_excel(file)
