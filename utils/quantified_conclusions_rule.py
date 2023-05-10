"""This module contains the class `QuantifiedConclusionsRule`"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import parse

from utils.preprocessing import LabelEncoderForColumns
from utils.rule import Rule


@dataclass(frozen=True, eq=False)
class QuantifiedConclusionsRule(Rule):
    """
    Represents a rule that contains a quantified conclusion but no quantification of the condition
    """

    @property
    def condition_string(self) -> str:
        return f'If you encounter {self.condition},'

    @staticmethod
    def from_string(
        string: str,
        label_encoder: LabelEncoderForColumns
    ) -> QuantifiedConclusionsRule:
        # Split rule
        comma_index = string.find(',')
        condition_string = string[:comma_index]
        conclusion_string = string[comma_index + 2:]

        # Parse splits
        args = Rule.parse_conclusion_string(conclusion_string, label_encoder)
        try:
            args["condition"] = parse.parse(
                "If you encounter {}", condition_string)[0]
        except Exception as exception:
            raise ValueError(f'not a valid condition: {condition_string}') from exception
        return QuantifiedConclusionsRule(**args)

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
    ) -> QuantifiedConclusionsRule:
        return super().from_relation(
            parameter,
            condition,
            parameter_type,
            param_values,
            relative_param_values,
            label_encoder,
            averaging_function,
        )

    @classmethod
    def merge_rules(
        cls,
        rules: List[QuantifiedConclusionsRule],
        label_encoder: LabelEncoderForColumns,
        averaging_function=np.nanmean,
        **kwargs
    ) -> QuantifiedConclusionsRule:
        return super().merge_rules(
            rules,
            label_encoder,
            averaging_function
        )
