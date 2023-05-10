"""This module contains the class `Rule`"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from distutils import util
from enum import unique, Enum
from typing import Any, Dict, Tuple, Union, Optional, List
import math
import numpy as np
import pandas as pd
import parse

from utils.preprocessing import LabelEncoderForColumns


@dataclass(frozen=True)
class Rule(ABC):
    """
    Represents a rule in the form of `if condition occurs execute conclusion`
    """

    PREDICTION_THRESHOLD = 0.30
    """The relative tolerance for two numerical values to be considered equal"""

    BIOU_THRESHOLD = 0.7
    """
    For lowlevel equality, the bound intersection-over-union of two intervals is calculated.
    BIoU values above this threshold are considered equal.
    """

    @unique
    class Action(Enum):
        """
        Enum that contains the different types of actions (parameter adjustments) 
        """
        UNDEFINED = 0,
        INCREASE_INCREMENTALLY = 1,
        """Increase the paramater incrementally by the step size"""
        DECREASE_INCREMENTALLY = 2,
        """Decrease the paramater incrementally by the step size"""
        ADJUST_INCREMENTALLY = 3,
        """Adjust the parameter incrementally without knowing the direction and step size"""
        SET = 4,
        """Set the parameter to the a specific value"""
        ADJUST_HARDWARE = 5,

        def __str__(self):
            return self.name.replace("_", " ").lower()

        @classmethod
        def from_string(cls, action: str):
            """Converts an action in form of a string to its enum counterpart"""
            action = action.strip().replace(" ", "_").upper()
            enum = cls[action]
            return enum

    @unique
    class ParamType(Enum):
        """Enum that contains the different types of parameter datatypes"""
        UNKNOWN = 0
        NUMERICAL = 1
        BOOLEAN = 2
        STRING = 3

    condition: str
    """What is the reason for a specific action"""

    parameter: str
    """Parameter that is adjusted"""

    parameter_type: Rule.ParamType
    """Datatype of the parameter"""

    label_encoder: LabelEncoderForColumns
    """Encoder used to encocer string values"""

    parameter_range: Optional[Tuple[float, float]] = None
    """Lower and upper limit for range of values to set the parameter (only numerical set action)"""

    absolute_mean_value: Optional[float] = None
    """Absolute mean value of the parameter in the graph (only numerical)"""

    relative_mean_value: Optional[float] = None
    """
    Step size to increase/decrease incrementally a certain parameter.
    (only numerical incremental action)
    """

    action_quantifier: Optional[Union[bool, int]] = None
    """Quantification of the action (only bool + string)"""

    action: Rule.Action = None
    """What to do with the cura parameter"""

    def __post_init__(self) -> None:
        """
        Sets the action if none was provided.

        For numerical rules relative adjustment actions are preferred.
        If no relative values are provided absolute adjustment or
        range adjustments actions are used.
        """
        if self.action is None:
            if self.relative_mean_value is not None and self.relative_mean_value > 0:
                action = Rule.Action.INCREASE_INCREMENTALLY
            elif self.relative_mean_value is not None and self.relative_mean_value < 0:
                action = Rule.Action.DECREASE_INCREMENTALLY
            elif self.parameter_range is not None and \
                    math.isclose(self.parameter_range[0], self.parameter_range[1]):
                action = Rule.Action.SET
            elif self.action_quantifier is not None:
                action = Rule.Action.SET
            else:
                action = Rule.Action.ADJUST_INCREMENTALLY
            super().__setattr__('action', action)

    def __str__(self) -> str:
        return f'{self.condition_string} {self.conclusion_string}'

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __eq__(self, other) -> bool:
        # TODO should this be replaced by a proper check on all properties?
        return Rule.lowlevel_conclusion_eq(self, other)

    @property
    @abstractmethod
    def condition_string(self) -> str:
        """String representation of the condition"""
        pass

    @property
    def conclusion_string(self) -> str:
        """String representation of the conclusion"""
        if self.relative_mean_value is not None and abs(self.relative_mean_value) > 0:
            target = f'by {round(abs(self.relative_mean_value), 2)}'
            # TODO XAI-607 enable after allowing mutliple conditions to get more meaningful ranges
            # if self.range_start != self.range_end:
            # additional_string = f' by {round(abs(self.step_size), 2)} in range {round(self.range_start, 2)} to {round(self.range_end, 2)}'
        elif self.action_quantifier is not None:
            if self.parameter_type == Rule.ParamType.STRING:
                target = f'to {self.label_encoder.inverse_transform(self.action_quantifier, self.parameter)}'
            else:
                target = f'to {self.action_quantifier}'
        elif self.parameter_range is not None:
            if math.isclose(self.parameter_range[0], self.parameter_range[1]):
                target = f'to {round(self.parameter_range[0], 2)}'
            else:
                target = f'in range {round(self.parameter_range[0], 2)} ' + \
                         f'to {round(self.parameter_range[1], 2)}'
        else:
            target = ""
        return f'try to {str(self.action)} the parameter {self.parameter} {target}'

    @staticmethod
    def highlevel_eq(rule_a: Rule, rule_b: Rule) -> bool:
        """Compares if two rules have the same parameter and condition"""
        return rule_a.parameter == rule_b.parameter and \
            rule_a.condition == rule_b.condition

    @staticmethod
    def midlevel_eq(rule_a: Rule, rule_b: Rule) -> bool:
        """Compares if two rules are highlevel equal and also contain the same action"""
        if not Rule.highlevel_eq(rule_a, rule_b):
            return False
        return rule_a.action == rule_b.action

    @staticmethod
    def lowlevel_conclusion_eq(rule_a: Rule, rule_b: Rule) -> bool:
        """
        Compares if the conclusions of two rules are lowlevel equal.

        - When comparing numerical values with other numerical values
          the relative distance must be < than `PREDICTION_THRESHOLD`
        - When comparing numerical values with intervals the value must
          either be inside the interval or the relative distance between
          it and one interval bound must be < than `PREDICTION_THRESHOLD / 2`
        - When comparing two numerical intervals the biou must be > than `BIOU_THRESHOLD`
        - When comparing string or boolean rules the action quantifiers must be equal
        """
        if not Rule.midlevel_eq(rule_a, rule_b):
            return False
        if not rule_b.is_low_level_conclusion_comparable() or \
           not rule_a.is_low_level_conclusion_comparable():
            return False

        distance = Rule.low_level_conclusion_distance(rule_a, rule_b)
        if rule_b.relative_mean_value not in [0, None] and \
           rule_a.relative_mean_value not in [0, None]:
            return distance < Rule.PREDICTION_THRESHOLD
        if rule_a.parameter_range is not None and rule_b.parameter_range is not None:
            rule_a_set = math.isclose(*rule_a.parameter_range)
            rule_b_set = math.isclose(*rule_b.parameter_range)
            if rule_a_set and rule_b_set:
                return distance < Rule.PREDICTION_THRESHOLD
            elif any([rule_a_set, rule_b_set]):
                return distance < Rule.PREDICTION_THRESHOLD / 2
            else:
                return distance > Rule.BIOU_THRESHOLD
        return distance == 0

    @staticmethod
    def lowlevel_eq(rule_a: Rule, rule_b: Rule) -> bool:
        """Compares if two rules are lowlevel equal"""
        return rule_a == rule_b

    def is_low_level_conclusion_comparable(self) -> bool:
        """
        if there is any possibility to compare the current rule to another on conclusion low level
        e.g. if the step size is in the same range this function should return True
        """
        if self.relative_mean_value is not None and abs(self.relative_mean_value) > 0:
            return True
        if self.absolute_mean_value is not None and abs(self.absolute_mean_value) > 0:
            return True
        if self.parameter_range is not None and not math.isclose(self.parameter_range[0], self.parameter_range[1]):
            # TODO XAI-558 range_start == range_end should return true
            return True
        if self.action_quantifier is not None:
            return True
        return False

    @staticmethod
    def bounded_intersection_over_union(interval_a: Tuple[float, float], interval_b: Tuple[float, float]) -> float:
        """Calculates the bound intersection-over-union for two intervals"""
        lower_bounds = [interval_b[0]]
        upper_bounds = [interval_b[1]]
        if not interval_a[0] <= interval_b[0] <= interval_a[1]:
            lower_bounds.append(interval_a[0])
        if not interval_a[0] <= interval_b[1] <= interval_a[1]:
            upper_bounds.append(interval_a[1])
        intersection = min(upper_bounds) - max(lower_bounds)
        union = max(upper_bounds) - min(lower_bounds)
        iou = intersection / union
        return iou

    @staticmethod
    def low_level_conclusion_distance(rule_a: Rule, rule_b: Rule) -> float:
        """Distance metric that defines how equal two conclusions are"""
        def v2v_distance(a: float, b: float) -> float:
            """Relative distance between two values"""
            a = abs(a)
            b = abs(b)
            # arithmetic mean change see https://en.wikipedia.org/wiki/Relative_change_and_difference, behaves analogously to what we defined so far
            # return (max(a, b)-min(a, b))/((a+b)/2)
            return 1 - min(a, b) / max(a, b)

        def v2r_distance(v: float, r: Tuple[float, float]) -> float:
            """Relative distance between a value and a range"""
            if r[0] <= v <= r[0]:
                return 0
            return min(v2v_distance(v, r[0]), v2v_distance(v, r[1]))

        if rule_a.relative_mean_value not in [0, None] and \
                rule_b.relative_mean_value not in [0, None]:
            return v2v_distance(rule_a.relative_mean_value, rule_b.relative_mean_value)
        elif rule_a.parameter_range is not None and rule_b.parameter_range is not None:
            groundtruth_set = math.isclose(*rule_a.parameter_range)
            prediction_set = math.isclose(*rule_b.parameter_range)
            if groundtruth_set and prediction_set:
                return v2v_distance(rule_a.parameter_range[0], rule_b.parameter_range[0])
            elif any([groundtruth_set, prediction_set]):
                value = rule_a.parameter_range[0] if groundtruth_set else rule_b.parameter_range[0]
                interval = rule_a.parameter_range if not groundtruth_set else rule_b.parameter_range
                return v2r_distance(value, interval)
            else:
                return Rule.bounded_intersection_over_union(
                    rule_a.parameter_range, rule_b.parameter_range)

        return 0 if rule_b.action_quantifier is not None \
            and rule_b.action_quantifier == rule_a.action_quantifier \
            else 1

    @staticmethod
    @abstractmethod
    def from_string(string: str, label_encoder: LabelEncoderForColumns) -> Rule:
        """Parses a string representation of a rule into a `Rule` object"""
        raise NotImplementedError(
            'from_string must be called from a inherited class and not from Rule itself'
        )

    @staticmethod
    def parse_conclusion_string(
        string: str,
        label_encoder: LabelEncoderForColumns
    ) -> Dict[str, Any]:
        """Parses the conclusion part of a rule string into its information"""
        result_incrementally = parse.parse(
            "try to {} the parameter {} by {}", string)
        result_range = parse.parse(
            "try to {} the parameter {} in range {} to {}", string)
        result_set = parse.parse("try to {} the parameter {} to {}", string)

        if result_incrementally is not None:
            parsed_action, parsed_parameter, parsed_step = result_incrementally
            categorical_rule, range_rule = False, False
        elif result_range is not None:
            parsed_action, parsed_parameter, parsed_range_start, parsed_range_end = result_range
            categorical_rule, range_rule = False, True
        elif result_set is not None:
            parsed_action, parsed_parameter, parsed_step = result_set
            categorical_rule, range_rule = True, False
        else:
            raise ValueError(f'Invalid conclusion string: {string}')

        rule_args = {
            "parameter": parsed_parameter,
            "label_encoder": label_encoder,
            "parameter_range": None,
            "absolute_mean_value": None,
            "relative_mean_value": None,
            "action_quantifier": None
        }

        try:
            rule_args["action"] = Rule.Action.from_string(parsed_action)
            if categorical_rule:
                try:
                    param_value = float(parsed_step)
                    rule_args["parameter_type"] = Rule.ParamType.NUMERICAL
                    rule_args["parameter_range"] = (param_value, param_value)
                except ValueError:
                    try:
                        # check if its a bool or string
                        rule_args["action_quantifier"] = util.strtobool(
                            parsed_step)
                        rule_args["parameter_type"] = Rule.ParamType.BOOLEAN
                    except ValueError:
                        rule_args["action_quantifier"] = label_encoder.transform(
                            parsed_step, parsed_parameter)
                        rule_args["parameter_type"] = Rule.ParamType.STRING
            elif range_rule:
                rule_args["parameter_type"] = Rule.ParamType.NUMERICAL
                rule_args["parameter_range"] = (
                    float(parsed_range_start), float(parsed_range_end))
            else:
                rule_args["parameter_type"] = Rule.ParamType.NUMERICAL
                if rule_args["action"] == Rule.Action.DECREASE_INCREMENTALLY:
                    rule_args["relative_mean_value"] = -1 * float(parsed_step)
                else:
                    rule_args["relative_mean_value"] = float(parsed_step)
        except Exception as exception:
            raise ValueError(
                f'not a valid conclusion: {string}') from exception

        return rule_args

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
    ) -> Rule:
        """
        Creates a rule from a Knowledge Graph relation, condition and parameter type

        Args:
            parameter (str): name of the cura parameter
            influence_key (str): name of the influence condition
            parameter_type (Rule.ParamType | None): type of the parameter (numerical/boolean/string)
            param_values (pd.Series): List of absolute values of parameters for experiments
            relative_param_values (pd.Series | None): List of relative values of parameters for experiments
            label_encoder (LabelEncoderForColumns): label encoder
            kwargs: Other arguments that should be passed to the constructor of the new rule

        Returns:
            Rule: new Rule object describing the KG Relation
        """

        if cls == Rule:
            raise NotImplementedError(
                'from_relation must be called from a inherited class and not from Rule itself'
            )

        rule_args = {
            "parameter": parameter,
            "parameter_type": parameter_type,
            "condition": condition,
            "label_encoder": label_encoder,
            "absolute_mean_value": averaging_function(param_values),
            "parameter_range": None,
            "relative_mean_value": None,
            "action_quantifier": None
        }

        if parameter_type == Rule.ParamType.NUMERICAL:
            p_std = np.std(param_values)
            rule_args['parameter_range'] = (
                rule_args['absolute_mean_value'] - p_std, rule_args['absolute_mean_value'] + p_std)
            rule_args['relative_mean_value'] = averaging_function(
                relative_param_values) if relative_param_values is not None else 0
        elif parameter_type == Rule.ParamType.BOOLEAN:
            rule_args['action'] = Rule.Action.SET
            # TODO XAI-558 this might be overly simplistic. In case of categorical values maybe it makes sense to take the previous value into consideration as a separate condition? In case of bools changed_parameters contains a propper difference
            if rule_args['absolute_mean_value'] >= 0.5:
                rule_args['action_quantifier'] = True
            else:
                rule_args['action_quantifier'] = False
        elif parameter_type == Rule.ParamType.STRING:
            rule_args['action'] = Rule.Action.SET
            # TODO XAI-558 this might be overly simplistic. In case of categorical values maybe it makes sense to take the previous value into consideration as a separate condition?
            # Note that changing it here also requires a change in merge_rules
            # simply keep the value that is occurring most often
            values, counts = np.unique(
                np.around(param_values), return_counts=True)
            ind = np.argmax(counts)
            try:
                rule_args['action_quantifier'] = values[ind]
            except IndexError:
                pass
        else:
            raise ValueError(
                f'Unknown parameter_type {parameter_type} encountered.')

        return cls(**rule_args, **kwargs)

    @classmethod
    def merge_rules(
        cls,
        rules: List[Rule],
        label_encoder: LabelEncoderForColumns,
        averaging_function=np.nanmean,
        **kwargs
    ) -> Rule:
        """
        Merge a list of rules pertaining to the same
        condition-parameterpair by averaging their values
        """

        if cls == Rule:
            raise NotImplementedError(
                'merge_rules must be called from a inherited class and not from Rule itself'
            )

        if len(rules) == 1:
            return rules[0]

        assert len({r.parameter for r in rules}
                   ) == 1, 'All rules must have the same parameter'
        assert len({r.condition for r in rules}
                   ) == 1, 'All rules must have the same condition'

        rule_args = {
            "parameter": rules[0].parameter,
            "parameter_type": rules[0].parameter_type,
            "condition": rules[0].condition,
            "label_encoder": label_encoder,
            "parameter_range": None,
            "absolute_mean_value": None,
            "relative_mean_value": None,
            "action_quantifier": None
        }

        range_starts, range_ends, abs_means, rel_means, quantifiers = [], [], [], [], []
        for rule in rules:
            if rule.parameter_range is not None:
                range_starts.append(rule.parameter_range[0])
                range_ends.append(rule.parameter_range[1])
            if rule.absolute_mean_value is not None:
                abs_means.append(rule.absolute_mean_value)
            if rule.relative_mean_value is not None:
                rel_means.append(rule.relative_mean_value)
            if rule.action_quantifier is not None:
                quantifiers.append(rule.action_quantifier)

        # Create merged rule
        if rule_args['parameter_type'] == Rule.ParamType.BOOLEAN:
            # special treatment for bools - if the mean is greater than 0.5 we treat it as true
            if averaging_function(quantifiers) >= 0.5:
                rule_args["action_quantifier"] = True
            else:
                rule_args["action_quantifier"] = False
        elif rule_args['parameter_type'] == Rule.ParamType.STRING:
            # special treatment for encoded strings - we keep the value occuring most often
            rule_args["action_quantifier"] = max(
                quantifiers, key=quantifiers.count)
        elif rule_args['parameter_type'] == Rule.ParamType.NUMERICAL:
            if len(range_starts) > 0:
                rule_args["parameter_range"] = (averaging_function(range_starts),
                                                averaging_function(range_ends))
            if len(abs_means) > 0:
                rule_args["absolute_mean_value"] = averaging_function(
                    abs_means)
            if len(rel_means) > 0:
                rule_args["relative_mean_value"] = averaging_function(
                    rel_means)
        else:
            raise ValueError(
                f"unknown Rule.ParamType {rule_args['parameter_type']} encountered")

        return cls(**rule_args, **kwargs)
