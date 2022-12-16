from __future__ import annotations

from typing import List, Tuple
from numpy import mean
import numpy as np
import math

from data_provider.synthetic_data_generation.types.pq_function import (
    GeneratedPQFunctions,
    PQFunction,
)
from utils.rule import Rule
from data_provider.synthetic_data_generation.types.pq_tuple import PQTuple
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig


class GeneratedPQ_Relations:

    relations: List[PQ_Relation]


class PQ_Relation:
    """This class brings PQ_functions from the Synthetic Data Generation and the
    rules extracted from the AIPE Dataset into a common format, which can then
    be translated from a KnowledgeGraphGenerator into a specific Relation
    """

    parameter: str
    quality: str
    conclusion_quantifications: float
    condition_scopes: List[List[float]]
    condition_values: List[float]
    action: str

    def __init__(
        self,
        parameter: str,
        quality: str,
        conclusion_quantifications: float,
        condition_scopes: List[List[float]],
        condition_values: List[float],
        action: str,
    ):
        self.parameter = parameter
        self.quality = quality
        self.conclusion_quantifications = conclusion_quantifications
        self.condition_scopes = condition_scopes
        self.condition_values = condition_values
        self.action = self.action_mapping(action)

    def action_mapping(self, action: Rule.Action):
        if action == Rule.Action.INCREASE_INCREMENTALLY:
            return "increase"
        elif action == Rule.Action.DECREASE_INCREMENTALLY:
            return "decrease"
        else: 
            return "implies"

    @staticmethod
    def from_rule(rule: Rule) -> PQ_Relation:
        """Generates a PQ_Relation from a rule

        Args:
            rule (Rule): based on rule

        Returns:
            PQ_Relation: generated relation
        """

        rel = PQ_Relation(
            parameter=rule.parameter,
            quality=rule.condition,
            conclusion_quantifications=rule.mean_value,
            condition_scopes=[[rule.condition_range[0], rule.condition_range[1]]],
            condition_values=[rule.mean_value],
            action=rule.action.value,
        )
        return rel

    @staticmethod
    def from_pq_function(
        pq_functions: GeneratedPQFunctions, pq_tuples: PQTuple, config: SdgConfig
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

            quality_config = config.get_quality_by_name(tuple[1])

            # Get the mean value over the relation
            changes = []
            p_old = func.inverse(quality_config.max_rating)
            for q_new in range(
                quality_config.max_rating - 1, quality_config.min_rating - 1, -1
            ):
                p_new = func.inverse(q_new, last_parameter=p_old)
                changes.append(p_new - p_old)
                p_old = p_new
            vmean = mean(changes)

            # Create the bins for the binned representations
            sample_points = PQ_Relation.sample_function(
                tuple[0], tuple[1], func, config
            )

            bin_edges = []
            bin_values = []

            for point in sample_points:
                bin_edges.append([point["start"], point["end"]])
                bin_values.append(point["value"])

            rels.append(
                PQ_Relation(
                    parameter=tuple[0],
                    quality=tuple[1],
                    conclusion_quantifications=vmean,
                    condition_scopes=bin_edges,
                    condition_values=bin_values,
                    action=Rule.Action.SET,
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
        arranged = np.arange(quality_config.min_rating, quality_config.max_rating + 1)
        samples = []
        for element in arranged:
            samples.append(pq_function.inverse(element))

        # Create bins
        edges = np.histogram_bin_edges(samples, bins=3)
        # Q = f(P) (for the bin edges)
        q_at_edges = pq_function(edges)

        results = []
        for i in range(0, len(q_at_edges) - 1):
            bin_start = q_at_edges[i]
            bin_end = q_at_edges[i + 1]
            # Sample between start and end
            # Get the sampling points i.e Intervall = 2.3 - 4 -> 2.3, 3.3

            sampling_points = []
            x = bin_start
            if bin_start > bin_end:
                while x >= bin_end:
                    sampling_points.append(x)
                    x -= 1
            else:
                while x <= bin_end:
                    sampling_points.append(x)
                    x += 1

            # Evaluate the function at the sampling points
            f_at_sampling_points = []
            for p in sampling_points:
                f_at_sampling_points.append(pq_function.inverse(p))

            bin_value = mean(f_at_sampling_points)
            bin_name = "µ_" + str(i) + ": " + str(bin_start) + "-" + str(bin_end)
            results.append(
                {
                    "name": bin_name,
                    "value": bin_value,
                    "start": bin_start,
                    "end": bin_end,
                }
            )

        return results
