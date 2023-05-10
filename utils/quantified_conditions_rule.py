"""This module contains the class `QuantifiedConditionsRule`"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np
import pandas as pd
import parse

from utils.preprocessing import LabelEncoderForColumns
from utils.rule import Rule


@dataclass(frozen=True, eq=False)
class QuantifiedConditionsRule(Rule):
    """
    Represents a rule that contains a quantified conclusion and a quantified condition
    """
    condition_range: Tuple[float, float] = None
    """Discretized bin in which the value of the condition lies"""

    def __post_init__(self) -> None:
        """Checks if the `condition_range` is defined"""
        if self.condition_range is None:
            raise ValueError('condition_range must be defined!')
        super().__post_init__()

    def __eq__(self, other) -> bool:
        # TODO should this be replaced by a proper check on all properties?
        return Rule.lowlevel_conclusion_eq(self, other) and \
            QuantifiedConditionsRule.lowlevel_condition_eq(self, other)

    def __hash__(self) -> int:
        return hash(self.__str__())

    @property
    def condition_string(self) -> str:
        return f'If you encounter {self.condition} in ' + \
               f'[{self.condition_range[0]}, {self.condition_range[1]}],'

    def is_low_level_condition_comparable(self) -> bool:
        """
        if there is any possibility to compare the current rule to another on condition low level
        """
        return self.condition_range[0] != self.condition_range[1] and \
            not any(math.isnan(c) for c in self.condition_range)

    @staticmethod
    def lowlevel_condition_eq(
        rule_a: QuantifiedConditionsRule,
        rule_b: QuantifiedConditionsRule
    ) -> bool:
        """Compares if the conditions of two rules are lowlevel equal"""
        if not Rule.midlevel_eq(rule_a, rule_b):
            return False
        if not isinstance(rule_a, QuantifiedConditionsRule) or not isinstance(rule_b, QuantifiedConditionsRule):
            return False
        if not rule_b.is_low_level_condition_comparable() or \
           not rule_a.is_low_level_condition_comparable():
            return False

        biou = QuantifiedConditionsRule.condition_range_biou(rule_a, rule_b)
        return math.isnan(biou) or biou > QuantifiedConditionsRule.BIOU_THRESHOLD

    @staticmethod
    def condition_range_iou(
        rule_a: QuantifiedConditionsRule,
        rule_b: QuantifiedConditionsRule
    ) -> float:
        """
        Calculates the intersection-over-union for the condition ranges of two rules
        """
        lower_bounds = [rule_a.condition_range[0], rule_b.condition_range[0]]
        upper_bounds = [rule_a.condition_range[1], rule_b.condition_range[1]]
        intersection = min(upper_bounds) - max(lower_bounds)
        union = max(upper_bounds) - min(lower_bounds)
        iou = intersection / union
        return iou

    @staticmethod
    def condition_range_biou(
        rule_a: QuantifiedConditionsRule,
        rule_b: QuantifiedConditionsRule
    ) -> float:
        """
        Calculates the bound intersection-over-union for the condition ranges of two rules.
        """
        return Rule.bounded_intersection_over_union(rule_a.condition_range, rule_b.condition_range)

    @staticmethod
    def from_string(string: str, label_encoder: LabelEncoderForColumns) -> QuantifiedConditionsRule:
        # Find second comma
        comma_index = string.find(',')
        comma_index = string.find(',', comma_index + 1)

        # Split rule
        condition_string = string[:comma_index]
        conclusion_string = string[comma_index + 2:]

        # Parse splits
        args = Rule.parse_conclusion_string(conclusion_string, label_encoder)
        try:
            args["condition"], lower_range, upper_range = parse.parse(
                "If you encounter {} in [{}, {}]", condition_string)
            args["condition_range"] = (float(lower_range), float(upper_range))
        except Exception as exception:
            raise ValueError(
                f'not a valid condition: {condition_string}') from exception
        return QuantifiedConditionsRule(**args)

    @classmethod
    def from_relation(
        cls,
        parameter: str,
        condition: str,
        parameter_type: Rule.ParamType,
        param_values: pd.Series,
        relative_param_values: pd.Series | None,
        label_encoder: LabelEncoderForColumns,
        averaging_function=np.nanmean,
        **kwargs
    ) -> QuantifiedConditionsRule:
        if 'condition_range' not in kwargs:
            raise ValueError('condition_range must be be provided!')
        return super().from_relation(
            parameter,
            condition,
            parameter_type,
            param_values,
            relative_param_values,
            label_encoder,
            averaging_function,
            condition_range=kwargs['condition_range']
        )

    @classmethod
    def merge_rules(
        cls,
        rules: List[QuantifiedConditionsRule],
        label_encoder: LabelEncoderForColumns,
        averaging_function=np.nanmean,
        **kwargs
    ) -> QuantifiedConditionsRule:
        condition_range_starts, condition_range_ends = [], []
        for rule in rules:
            condition_range_starts.append(rule.condition_range[0])
            condition_range_ends.append(rule.condition_range[1])
        condition_range = (averaging_function(condition_range_starts),
                           averaging_function(condition_range_ends))
        return super().merge_rules(
            rules,
            label_encoder,
            averaging_function,
            condition_range=condition_range
        )
