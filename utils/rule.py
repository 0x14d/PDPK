from __future__ import annotations

import math
from distutils import util
from enum import unique, Enum
from typing import Union

import numpy as np
import pandas as pd
import parse

from utils.preprocessing import LabelEncoderForColumns


class Rule:

    CONDITION_THRESHOLD = 0.9
    # for equality, the intersection-over-union of the condition ranges is calculated.\
    # IoU values above this threshold are considered equal
    PREDICTION_THRESHOLD = 0.30
    # the relative tolerance for two numerical values to be considered equal

    @unique
    class Action(Enum):
        UNDEFINED = 0,
        INCREASE_INCREMENTALLY = 1,
        DECREASE_INCREMENTALLY = 2,
        ADJUST_INCREMENTALLY = 3,
        SET = 4,
        ADJUST_HARDWARE = 5,

        def __str__(self):
            return self.name.replace("_", " ").lower()

        @classmethod
        def from_string(cls, action: str):
            action = action.strip().replace(" ", "_").upper()
            enum = cls[action]
            return enum

    @unique
    class ParamType(Enum):
        UNKNOWN = 0
        NUMERICAL = 1
        BOOLEAN = 2
        STRING = 3

    def __init__(self, label_encoder: LabelEncoderForColumns):
        # TODO wouldn't it be better to initialze everything in the constructor already? make sure everythings set up correctly? \
        # including parameter type selection? or using a Builder pattern
        self.condition = str()  # what is the reason for a specific action
        # discretized bin in which the value of the condition lies
        self.condition_range: tuple[float, float] = (np.nan, np.nan)
        self.action = Rule.Action.UNDEFINED  # what to do with the cura parameter
        self.parameter = str()  # cura parameter
        self.mean_value = float()  # mean value of the parameter in the graph
        # lower limit for range of values to set the parameter
        self.range_start: float = float()
        self.range_end = float()  # upper limit for range of values to set the parameter
        # TODO XAI-558 is it possible to combine step_size and action_quantifier? (probably easier if we use inheritance)
        # We have two options here: leave it as float/bool and transform to string in tostr or place allow string values -> currently the first is implemented
        self.action_quantifier: Union[None, bool, int] = None
        # step size to increase/decrease incrementally a certain parameter
        self.step_size = float()
        # type of the rule TODO XAI-587 does it make sense to use inheritance to handle different behaviour for categorical/numerical rules?
        self.parameter_type: Rule.ParamType | None = None
        self._label_encoder: LabelEncoderForColumns = label_encoder

    def __eq__(self, other):
        # TODO should this be replaced by a proper check on all properties?
        return Rule.lowlevel_eq(self, other)

    def __hash__(self):
        return hash(self.__str__())

    def __str__(self):
        string = ""
        if math.isnan(self.step_size) and math.isnan(self.range_start) and math.isnan(self.range_end):
            string = ""
        elif abs(self.step_size) > 0:
            string = f' by {round(abs(self.step_size), 2)}'
            # TODO XAI-607 enable after allowing mutliple conditions to get more meaningful ranges
            # if self.range_start != self.range_end:
            # additional_string = f' by {round(abs(self.step_size), 2)} in range {round(self.range_start, 2)} to {round(self.range_end, 2)}'
        elif self.action_quantifier is not None:
            if self.parameter_type == Rule.ParamType.STRING:
                string = f' to {self._label_encoder.inverse_transform(self.action_quantifier, self.parameter)}'
            else:
                string = f' to {self.action_quantifier}'
        else:
            if math.isclose(self.range_start, self.range_end):
                string = f' to {round(self.range_start, 2)}'
            else:
                string = f' in range {round(self.range_start, 2)} to {round(self.range_end, 2)}'
        additional_string = string

        return f"If you encounter {self.condition} in [{self.condition_range[0]}, {self.condition_range[1]}], try to {str(self.action)} the " \
               f"parameter {self.parameter}" + additional_string

    def is_given_relation(self, influence, parameter):
        if influence == self.condition and parameter == self.parameter:
            return True
        else:
            return False

    def is_low_level_comparable(self):
        """
        if there is any possibility to compare the current rule to another on low level e.g. if the step size is in the same range this function should return True
        """
        if abs(self.step_size) > 0:
            return True
        elif not math.isclose(self.range_start, self.range_end):
            # TODO XAI-558 range_start == range_end should return true
            return True
        elif not (self.action_quantifier is None):
            return True
        elif self.condition_range[0] != self.condition_range[1] and self.condition_range != np.nan:
            return True
        else:
            return False

    @classmethod
    def from_string(cls, string: str, label_encoder: LabelEncoderForColumns):
        rule = cls(label_encoder)
        result = parse.parse(
            "If you encounter {} in [{}, {}], try to {} the parameter {} by {}", string)
        categorical_rule = False
        range_rule = False
        if result is None:
            range_rule = True
            result = parse.parse(
                "If you encounter {} in [{}, {}], try to {} the parameter {} in range {} to {}", string)
            if result is None:
                categorical_rule = True
                range_rule = False
                parsed_condition, parsed_condition_lower, parsed_condition_upper, parsed_action, parsed_parameter, parsed_step = parse.parse(
                    "If you encounter {} in [{}, {}], try to {} the parameter {} to {}", string)

            else:
                parsed_condition, parsed_condition_lower, parsed_condition_upper, parsed_action, parsed_parameter, parsed_range_start, parsed_range_end = result
        else:
            parsed_condition, parsed_condition_lower, parsed_condition_upper, parsed_action, parsed_parameter, parsed_step = result

        try:
            rule.condition = parsed_condition
            rule.condition_range = (
                float(parsed_condition_lower), float(parsed_condition_upper))

            try:
                rule.action = Rule.Action.from_string(parsed_action)
            except KeyError as e:
                raise KeyError(
                    'parsing failed - not a validly formatted rule: ' + str(e))
            rule.parameter = parsed_parameter
            if range_rule is False:
                if categorical_rule is True:
                    try:
                        # check whether it's a pseudo categorical rule (happens if there is no intervall but all values are the same)
                        param_value = float(parsed_step)
                        rule.parameter_type = Rule.ParamType.NUMERICAL
                        rule.range_start = param_value
                        rule.range_end = param_value
                    except ValueError:
                        try:
                            # check if its a bool or string
                            rule.action_quantifier = util.strtobool(
                                parsed_step)
                            rule.parameter_type = Rule.ParamType.BOOLEAN
                        except ValueError:
                            rule.action_quantifier = label_encoder.transform(
                                parsed_step, rule.parameter)
                            rule.parameter_type = Rule.ParamType.STRING
                else:
                    rule.parameter_type = Rule.ParamType.NUMERICAL
                    if rule.action == Rule.Action.DECREASE_INCREMENTALLY:
                        rule.step_size = -1 * float(parsed_step)
                    else:
                        rule.step_size = float(parsed_step)
            else:
                rule.parameter_type = Rule.ParamType.NUMERICAL
                rule.range_start = float(parsed_range_start)
                rule.range_end = float(parsed_range_end)

        except Exception as e:
            raise ValueError(f'not a valid rule: {string} - {e}')
        return rule

    @classmethod
    def from_relation(cls,
                      parameter_key: str,
                      influence_key: str,
                      parameter_type: Rule.ParamType | None,
                      param_values: pd.Series,
                      relative_param_values: pd.Series | None,
                      influence_bin: tuple[float, float] | None,
                      label_encoder: LabelEncoderForColumns,
                      averaging_function=np.nanmean) -> Rule:
        """
        create rule from Knowledge Graph relation, condition and parameter type

        Args:
            parameter_key (str): name of the cura parameter
            influence_key (str): name of the influence condition
            parameter_type (Rule.ParamType | None): type of the parameter (numerical/boolean/string)
            param_values (pd.Series): List of absolute values of parameters for experiments
            relative_param_values (pd.Series | None): List of relative values of parameters for experiments
            influence_bin (tuple[float, float] | None): tuple of lower, upper bin edges in which the influence value resides
            label_encoder (LabelEncoderForColumns): label encoder

        Returns:
            Rule: new Rule object describing the KG Relation
        """

        rule = cls(label_encoder)
        rule.parameter = parameter_key
        rule.condition = influence_key
        rule.parameter_type = parameter_type

        mean_param_values = averaging_function(param_values)
        mean_relative_param_values = averaging_function(
            relative_param_values) if relative_param_values is not None else 0

        rule.mean_value = mean_param_values
        if rule.parameter_type == Rule.ParamType.NUMERICAL:
            param_std = np.std(param_values)
            rule.range_start = mean_param_values - param_std
            rule.range_end = mean_param_values + param_std
            rule.mean_value = mean_param_values
            rule.step_size = mean_relative_param_values
            rule.set_action()
        elif rule.parameter_type == Rule.ParamType.BOOLEAN:
            rule.action = Rule.Action.SET
            # TODO XAI-558 this might be overly simplistic. In case of categorical values maybe it makes sense to take the previous value into consideration as a separate condition? In case of bools changed_parameters contains a propper difference
            if mean_param_values >= 0.5:
                rule.action_quantifier = True
            else:
                rule.action_quantifier = False
        elif rule.parameter_type == Rule.ParamType.STRING:
            rule.action = Rule.Action.SET
            # TODO XAI-558 this might be overly simplistic. In case of categorical values maybe it makes sense to take the previous value into consideration as a separate condition?
            # Note that changing it here also requires a change in merge_rules
            # simply keep the value that is occurring most often
            values, counts = np.unique(
                np.around(param_values), return_counts=True)
            ind = np.argmax(counts)
            try:
                rule.action_quantifier = values[ind]
            except IndexError:
                rule.action_quantifier = None
        else:
            raise ValueError(
                f'Unknown parameter_type {rule.parameter_type} encountered.')

        if influence_bin is not None:
            lower_bin_edge, upper_bin_edge = influence_bin
            rule.condition_range = (lower_bin_edge, upper_bin_edge)

        return rule

    @staticmethod
    def highlevel_eq(baseline_rule: Rule, prediction_rule: Rule) -> bool:
        if baseline_rule.parameter == prediction_rule.parameter and baseline_rule.condition == prediction_rule.condition:
            return True
        return False

    @staticmethod
    def midlevel_eq(baseline_rule: Rule, prediction_rule: Rule) -> bool:
        if Rule.highlevel_eq(baseline_rule, prediction_rule):
            if baseline_rule.action == prediction_rule.action:
                return True
        return False

    @classmethod
    def lowlevel_eq(cls, baseline_rule: Rule, prediction_rule: Rule, verbose=False) -> bool:

        if Rule.midlevel_eq(baseline_rule, prediction_rule):
            if prediction_rule.is_low_level_comparable() is True and baseline_rule.is_low_level_comparable():
                iou = Rule.condition_range_iou(baseline_rule, prediction_rule)
                if np.isnan(iou) or iou > cls.CONDITION_THRESHOLD:
                    # TODO XAI-558 add case for range_start == range_end
                    if prediction_rule.mean_value:
                        if baseline_rule.range_end > prediction_rule.mean_value > baseline_rule.range_start:
                            return True
                    if baseline_rule.mean_value:
                        if prediction_rule.range_end > baseline_rule.mean_value > prediction_rule.range_start:
                            return True
                        # TODO XAI-558 use None instead of 0
                    if prediction_rule.step_size != 0 and baseline_rule.step_size != 0:
                        if math.isclose(baseline_rule.step_size, prediction_rule.step_size, rel_tol=cls.PREDICTION_THRESHOLD) is True:
                            return True
                    if prediction_rule.action_quantifier is not None and prediction_rule.action_quantifier == baseline_rule.action_quantifier:
                        return True
                if verbose:
                    print(f'{prediction_rule} is not equal {baseline_rule}')
        return False

    @staticmethod
    def condition_range_iou(rule_a: Rule, rule_b: Rule) -> float:
        lower_bounds = [rule_a.condition_range[0], rule_b.condition_range[0]]
        upper_bounds = [rule_a.condition_range[1], rule_b.condition_range[1]]
        intersection = min(upper_bounds) - max(lower_bounds)
        union = max(upper_bounds) - min(lower_bounds)
        iou = intersection / union
        return iou

    # TODO XAI-636 may be unset as long as action is publicly accessible
    def set_action(self) -> None:
        """
        Sets the rule action based on values of step_size and ranges
        :return:
        """
        if self.step_size > 0:
            self.action = Rule.Action.INCREASE_INCREMENTALLY
        elif self.step_size < 0:
            self.action = Rule.Action.DECREASE_INCREMENTALLY
        else:
            if math.isclose(self.range_start, self.range_end):
                self.action = Rule.Action.SET
            elif self.action_quantifier is not None:
                self.action = Rule.Action.SET
            else:
                self.action = Rule.Action.ADJUST_INCREMENTALLY

    @staticmethod
    def merge_rules(rules: list[Rule], label_encoder: LabelEncoderForColumns, averaging_function=np.nanmean) -> Rule:
        """merge a list of rules pertaining to the same condition-parameter pair by averaging their values"""
        if len(rules) == 1:
            return rules[0]
        else:
            range_starts, range_ends, means, step_sizes, quantifiers = [], [], [], [], []
            condition_range_starts, condition_range_ends = [], []
            for rule in rules:
                range_starts.append(rule.range_start)
                range_ends.append(rule.range_end)
                condition_range_starts.append(rule.condition_range[0])
                condition_range_ends.append(rule.condition_range[1])
                means.append(rule.mean_value)
                if rule.step_size is not None:
                    step_sizes.append(rule.step_size)
                if rule.action_quantifier is not None:
                    quantifiers.append(rule.action_quantifier)

            merged_rule = Rule(label_encoder)
            merged_rule.parameter = rules[0].parameter
            merged_rule.condition = rules[0].condition
            merged_rule.condition_range = (averaging_function(
                condition_range_starts), averaging_function(condition_range_ends))
            merged_rule.parameter_type = rules[0].parameter_type

            if merged_rule.parameter_type == Rule.ParamType.BOOLEAN:
                # special treatment for bools - if the mean is greater than 0.5 we treat it as true
                if averaging_function(quantifiers) >= 0.5:
                    merged_rule.action_quantifier = True
                else:
                    merged_rule.action_quantifier = False
            elif merged_rule.parameter_type == Rule.ParamType.STRING:
                # special treatment for encoded strings - we keep the value occuring most often
                merged_rule.action_quantifier = max(
                    quantifiers, key=quantifiers.count)
            elif merged_rule.parameter_type == Rule.ParamType.NUMERICAL:
                merged_rule.range_start = averaging_function(range_starts)
                merged_rule.range_end = averaging_function(range_ends)
                merged_rule.mean_value = averaging_function(means)
                merged_rule.step_size = averaging_function(step_sizes)
                merged_rule.quantifiers = averaging_function(quantifiers)
            else:
                raise ValueError(
                    f"unknown Rule.ParamType {merged_rule.parameter_type} encountered")

            merged_rule.set_action()
            return merged_rule

    @property
    def label_encoder(self):
        return self._label_encoder
