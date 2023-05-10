from __future__ import annotations

from enum import Enum
import itertools
from typing import List, Union, Type, Optional
from numpy import mean
import numpy as np

from data_provider.synthetic_data_generation.types.pq_function import (
    GeneratedPQFunctions,
    PQFunction,
)
from utils.preprocessing import LabelEncoderForColumns
from utils.rule import Rule
from utils.quantified_conclusions_rule import QuantifiedConclusionsRule
from utils.quantified_conditions_rule import QuantifiedConditionsRule
from data_provider.synthetic_data_generation.types.pq_tuple import GeneratedPQTuples
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig


class GeneratedPQ_Relations:

    relations: List[PQ_Relation]


class PQ_Relation:
    """This class brings PQ_functions from the Synthetic Data Generation and the
    rules extracted from the AIPE Dataset into a common format, which can then
    be translated from a KnowledgeGraphGenerator into a specific Relation
    """

    class Action(str, Enum):
        ADJUST = 'adjust',
        SET = 'set'

        @staticmethod
        def from_rule_action(action: Rule.Action) -> PQ_Relation.Action:
            if action == Rule.Action.SET:
                return PQ_Relation.Action.SET
            return PQ_Relation.Action.ADJUST

    parameter: str
    quality: str
    condition_scopes: List[List[float]]
    parameter_values_absolute: List[float]
    parameter_values_relative: List[float]
    action: PQ_Relation.Action

    def __init__(
        self,
        parameter: str,
        quality: str,
        condition_scopes: List[List[float]],
        parameter_values_absolute: List[float],
        parameter_values_relative: List[float],
        action: Rule.Action,
    ):
        self.parameter = parameter
        self.quality = quality
        self.condition_scopes = condition_scopes
        self.parameter_values_absolute = parameter_values_absolute
        self.parameter_values_relative = parameter_values_relative
        self.action = PQ_Relation.Action.from_rule_action(action)

    @property
    def conclusion_quantifications(self) -> List[float]:
        """
        List of all parameter values
        (absolute or relative depending on the action)
        """
        if self.action == PQ_Relation.Action.ADJUST:
            return self.parameter_values_relative
        if self.action == PQ_Relation.Action.SET:
            return self.parameter_values_absolute

    @property
    def conclusion_quantification_mean(self) -> float:
        """
        Mean of all parameter values
        (absolute or relative depending on the action)
        """
        return mean(self.conclusion_quantifications)

    @property
    def quantified_conclusion_prefix(self) -> str:
        """String describing if the quantified conclusion is relative or absolute"""
        if self.action == PQ_Relation.Action.ADJUST:
            return 'relatively'
        if self.action == PQ_Relation.Action.SET:
            return 'absolutely'

    def to_rules(
        self,
        label_encoder: LabelEncoderForColumns,
        rule_class: Union[Type[QuantifiedConditionsRule],
                          Type[QuantifiedConclusionsRule]]
    ) -> List[Rule]:
        """
        Converts the relation to a list of rules.
        Each rule represents a quality bin.
        """
        # INCREASE / DECREASE action
        if rule_class is QuantifiedConditionsRule:
            rules = [
                rule_class.from_relation(
                    parameter=self.parameter,
                    condition=self.quality,
                    parameter_type=Rule.ParamType.NUMERICAL,
                    param_values=[parameter_value_absolute],
                    relative_param_values=[parameter_value_relative],
                    label_encoder=label_encoder,
                    condition_range=tuple(quality_scope)
                )
                for quality_scope, parameter_value_relative, parameter_value_absolute
                in zip(self.condition_scopes, self.parameter_values_relative,
                       self.parameter_values_absolute)
            ]
        # TODO ADELS-466 meaning the means is not the propper way to go, Synthetic Data Provider should now whether QuantifiedConditionRules are generated and only then split the conditions
        elif rule_class is QuantifiedConclusionsRule:
            rules = [
                rule_class.from_relation(
                    parameter=self.parameter,
                    condition=self.quality,
                    parameter_type=Rule.ParamType.NUMERICAL,
                    param_values=np.mean(self.parameter_values_absolute),
                    relative_param_values=np.mean(
                        self.parameter_values_relative),
                    label_encoder=label_encoder
                )]
        return rules

    @staticmethod
    def from_rule(rule: Rule) -> Optional[PQ_Relation]:
        """Generates a PQ_Relation from a rule

        Args:
            rule (Rule): based on rule

        Returns:
            PQ_Relation: generated relation, None if rule isn't numerical
        """
        # TODO: Handle non numerical rules
        if rule.parameter_type != Rule.ParamType.NUMERICAL:
            return None

        if isinstance(rule, QuantifiedConditionsRule):
            condition_scopes = [
                [rule.condition_range[0], rule.condition_range[1]]]
        else:
            condition_scopes = []

        rel = PQ_Relation(
            parameter=rule.parameter,
            quality=rule.condition,
            condition_scopes=condition_scopes,
            parameter_values_absolute=[rule.absolute_mean_value],
            parameter_values_relative=[rule.relative_mean_value],
            action=rule.action,
        )
        return rel

    @staticmethod
    def from_pq_function(
        pq_functions: GeneratedPQFunctions, pq_tuples: GeneratedPQTuples, config: SdgConfig
    ) -> List[PQ_Relation]:
        """Generates the relations from GeneratedPQFunctions

        Args:
            pq_functions (GeneratedPQFunctions): the PQFunctions to generate the
                rules from

        Returns:
            List[PQ_Relation]: PQRelations representing the PQFunctions
        """

        rels = []
        for tuple in pq_tuples.expert_knowledge:
            func = pq_functions.pq_functions[tuple]

            # Create the bins for the binned representations
            sample_points = PQ_Relation.sample_function(
                tuple[0], tuple[1], func, config
            )

            bin_edges = []
            bin_values_absolute = []
            bin_values_relative = []

            for point in sample_points:
                bin_edges.append([point["start"], point["end"]])
                bin_values_absolute.append(point["value_absolute"])
                bin_values_relative.append(point["value_relative"])

            rels.append(
                PQ_Relation(
                    parameter=tuple[0],
                    quality=tuple[1],
                    condition_scopes=bin_edges,
                    parameter_values_absolute=bin_values_absolute,
                    parameter_values_relative=bin_values_relative,
                    action=Rule.Action.ADJUST_INCREMENTALLY,
                )
            )
        return rels

    def sample_function(
        parameter: str, quality: str, pq_function: PQFunction, config: SdgConfig
    ):
        """Samples the relation over a number of bins

        Args:
            parameter (str): parameter of the relation
            quality (str): quality of the relation

        Returns:
            List(dict): List of dictionaries containing the bins
        """
        quality_config = config.get_quality_by_name(quality)

        # Sample the array
        arranged = np.arange(quality_config.min_rating,
                             quality_config.max_rating + 1)
        samples = []
        for element in arranged:
            samples.append(pq_function.inverse(element))

        # Create bins
        edges = np.histogram_bin_edges(samples, bins=3)
        # Q = f(P) (for the bin edges)
        q_at_edges = pq_function(edges)

        results = []
        for i in range(0, len(q_at_edges) - 1):
            bin_start = min(q_at_edges[i], q_at_edges[i + 1])
            bin_end = max(q_at_edges[i], q_at_edges[i + 1])
            # Sample between start and end
            # Get the sampling points i.e Intervall = 2.3 - 4 -> 2.3, 3.3

            sampling_points = []
            x = bin_start
            while x <= bin_end:
                sampling_points.append(x)
                x += 1

            # Evaluate the function at the sampling points
            f_at_sampling_points = []
            for p in sampling_points:
                f_at_sampling_points.append(pq_function.inverse(p))

            bin_value_relative = mean(
                [
                    after - before
                    for after, before in zip(f_at_sampling_points[:-1], f_at_sampling_points[1:])
                ]
            )
            bin_value = mean(f_at_sampling_points)
            bin_name = "µ_" + str(i) + ": " + \
                str(bin_start) + "-" + str(bin_end)
            results.append(
                {
                    "name": bin_name,
                    "value_absolute": bin_value,
                    "value_relative": bin_value_relative,
                    "start": bin_start,
                    "end": bin_end,
                }
            )

        return results
