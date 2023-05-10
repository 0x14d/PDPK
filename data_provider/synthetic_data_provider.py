"""
This module provides the class `SyntheticDataProvider`
"""

# pylint: disable=too-many-arguments, no-self-use, unused-argument, unused-variable, too-many-locals

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import uuid

import pandas as pd
from utilities import to_camel_case

from data_provider.synthetic_data_generation.config.modules.pq_tuple_generator_config \
    import BasicPQTupleGeneratorConfig
from data_provider.synthetic_data_generation.config.sdg_config import SdgConfig
from data_provider.synthetic_data_generation.types.experiments \
    import GeneratedDataset, GeneratedExperiment, GeneratedExperimentSeries
from .synthetic_data_generation import database_experiment_interfaces as dei
from .abstract_data_provider import AbstractDataProvider
from .synthetic_data_generation.synthetic_data_generator import SyntheticDataGenerator


class SyntheticDataProvider(AbstractDataProvider):
    """
    Class that acts as interface between the AIPE dataset usage and the synthetic data generation.

    It provides funtionality to generate synthetic experiments and parses them into the AIPE format.
    """

    data_generator: SyntheticDataGenerator
    generated_experiments: Optional[List[dei.ExperimentFromDb]]

    def __init__(self, config: Optional[Union[SdgConfig, str, dict]] = None) -> None:
        """
        Parameters:
            config (SdgConfig | str | dict | None): Configuration of the synthetic data generator
        """
        super().__init__()

        if config is None:
            # TODO: Add default config
            config = SdgConfig(
                parameters=[
                    'adhesion_type', 'bottom_layers', 'bridge_fan_speed', 'bridge_settings_enabled',
                    'brim_line_count', 'cool_fan_enabled', 'cool_fan_full_at_height',
                    'cool_fan_speed', 'flow_rate_extrusion_offset_factor', 'infill_sparse_density',
                    'infill_wall_line_count', 'ironing_enabled', 'ironing_only_highest_layer',
                    'layer_height', 'limit_support_retractions', 'material_bed_temperature',
                    'material_bed_temperature_layer_0', 'material_final_print_temperature',
                    'material_flow', 'material_initial_print_temperature',
                    'material_print_temperature','material_print_temperature_layer_0',
                    'retraction_amount', 'retraction_count_max', 'retraction_enable',
                    'retraction_extrusion_window', 'retraction_hop', 'retraction_hop_enabled',
                    'retraction_min_travel', 'retraction_retract_speed', 'retraction_speed',
                    'slicing_tolerance', 'small_feature_speed_factor', 'speed_infill',
                    'speed_ironing', 'speed_layer_0', 'speed_print', 'speed_print_layer_0',
                    'speed_support_roof', 'speed_topbottom', 'speed_travel', 'speed_travel_layer_0',
                    'speed_wall', 'speed_wall_0', 'speed_wall_x', 'support_enable',
                    'support_initial_layer_line_distance', 'support_interface_enable',
                    'support_top_distance', 'support_tree_angle', 'support_tree_branch_diameter',
                    'support_tree_branch_diameter_angle', 'support_tree_collision_resolution',
                    'support_tree_enable', 'support_type', 'support_wall_count',
                    'support_xy_distance_overhang', 'support_z_distance', 'top_layers',
                    'travel_avoid_distance', 'wall_line_count'
                ],
                qualities=[
                    'blobs', 'burning', 'gaps', 'layer_misalignment', 'layer_separation',
                    'line_misalignment', 'lost_adhesion', 'not_labelable', 'over_extrusion',
                    'overall_ok', 'poor_bridging', 'stringing', 'under_extrusion', 'warping'
                ],
                pq_tuple_generator=BasicPQTupleGeneratorConfig(
                    num_parameters=46,
                    num_qualities=14
                )
            )

        self.data_generator = SyntheticDataGenerator(config)
        self.generated_experiments = None
    
    def get_executed_experiments_data(
        self,
        rating_averaging: str | Callable = 'mean',
        measurement_averaging: str | Callable = 'median',
        include_oneoffs: str | bool = False,
        printer: str | None = None,
        experiment_series_id: str | None = None,
        completed_only: bool = False,
        labelable_only: bool = True,
        containing_insights_only: bool = False,
        material_type: str | None = None,
        limit: int | None = None
    ) -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, set[Any]],
               set[str], list[dei.ExperimentDict], dei.ExperimentSeries]:
        df, lok, lov, boolean_parameters, experiments, experiment_series = super().get_executed_experiments_data(
            rating_averaging=rating_averaging,
            measurement_averaging=measurement_averaging,
            include_oneoffs=include_oneoffs,
            printer=printer,
            experiment_series_id=experiment_series_id,
            completed_only=completed_only,
            labelable_only=labelable_only,
            containing_insights_only=containing_insights_only,
            material_type=material_type,
            limit=limit
        )
        # By default only the changed parameters are included
        # -> We want to include all parameters
        lok['process_parameters'] = set(self.data_generator.pq_tuples.selected_parameters)
        return df, lok, lov, boolean_parameters, experiments, experiment_series

    def _parse_experiment(self, experiment, process_parameters,
                          rating_aggregation_func, measure_aggregation_func):
        # By default only the changed parameters are included
        # -> We want to include all parameters
        return super()._parse_experiment(
            experiment=experiment,
            process_parameters=set(self.data_generator.pq_tuples.selected_parameters),
            rating_aggregation_func=rating_aggregation_func,
            measure_aggregation_func=measure_aggregation_func
        )      

    def _prepare_experiment_data(
            self, completed_only, containing_insights_only, experiment_series_id,
            include_oneoffs, labelable_only, limit, material_type, printer
        ) -> Tuple[dei.ExperimentSeries, List[dei.ExperimentDict]]:
        # TODO XAI-652: add filtering method to support method arguments (params, query, limit)
        # TODO XAI-652: Exp Limit, Params to Include
        # TODO XAI-652: filter the columns to only include those params

        params_to_include = self._default_experiment_attributes_to_fetch + \
                            ['insights', 'stl_file_id']

        # get data from data generator
        if self.generated_experiments is None:
            logging.info("generating new dataset")
            dataset = self.data_generator.dataset
            self.generated_experiments = self._parse_synthetic_dataset(dataset)

        # parse back into dict to use legacy code
        experiments = [exp.to_dict() for exp in self.generated_experiments]

        def get_series_dict_index(exp_list: List[dei.ExperimentFromDb]) -> Dict[str, list[int]]:
            series_dict = defaultdict(list)
            for exp in exp_list:
                series_dict[exp.series_id].append(exp.id)
            return series_dict

        if containing_insights_only:
            self._fix_not_labelable(experiments)
        # removed "filtered_for_insights", since all Experiments include insights (as per interface)
        # "Last rating influences" might be null though.

        experiment_series = get_series_dict_index(self.generated_experiments)
        # _, experiment_series = self._determine_experiment_series_for_one_offs(experiments_with_insights)
        # TODO XAI-652: fix "determine series for one offs" or replace ?
        return experiment_series, experiments

    def _filter_experiment_data(
            self,
            experiments: Iterable[dei.ExperimentFromDb],
            include_oneoffs=False,
            printer=None,
            experiment_series_id=None,
            completed_only=False,
            labelable_only=True,
            containing_insights_only=False,
            material_type=None,
            limit=None
        ) -> List[dei.ExperimentFromDb]:
        # TODO XAI-652: for now, just "accept" all the son_query keywords, will have to actually handle them later on
        # TODO XAI-652: for once, not sure if this is needed anyway. If it is, this setup maybe should happen before the generation, instead of filtering afterwards? but afterwards gives more traceability
        return list(experiments) if not isinstance(experiments,
                                                   list) else experiments

    def _parse_synthetic_dataset(self, dataset: GeneratedDataset) -> List[dei.ExperimentFromDb]:
        """
        Parses a synthetic dataset to a list of AIPE experiments

        Parameters:
            dataset (GeneratedDataset): Synthetic dataset

        Returns:
            List of all generated experiments parsed to the AIPE format
        """
        parsed_experiments: List[dei.ExperimentFromDb] = []

        for experiment_series in dataset.experiment_series:
            for experiment in experiment_series.experiments:
                parsed_experiment = self._parse_synthetic_experiment(experiment, experiment_series)
                parsed_experiments.append(parsed_experiment)

        return parsed_experiments

    def _parse_synthetic_experiment(
        self,
        experiment: GeneratedExperiment,
        experiment_series: GeneratedExperimentSeries,
    ) -> dei.ExperimentFromDb:
        """
        Parses a synthetic experiment to the AIPE format

        Parameters:
            experiment (GeneratedExperiment): Synthetic experiment
            experiment_series (GeneratedExperimentSeries): Experiment series that contains the
                experiment

        Returns:
            Experiment parsed to the AIPE format
        """
        ParamType = Union[float, str, int, bool]
        LRIDictType = Dict[str, Dict[str, ParamType]]

        _stl_id = f"sdg_stl_file_{self.get_random_id()}"
        _series_id = f"sdg_series_{experiment_series.experiment_series_id}"
        _user_id = f"sdg_user_{self.get_random_id()}"
        _experiment_id = experiment.experiment_id
        _now = datetime.now()
        _material = dei.Material.from_dict(experiment.to_dict())

        rating_dict = experiment.qualities.copy()
        rating_dict['user_id'] = _user_id
        rating_dict['rating_date'] = _now
        rating_dict['comment'] = "created by SDG"
        _rating = dei.Rating.from_dict(rating_dict)

        # mock some environment measurements
        _measurements = [
            dei.Measurement(humidity=42., temperature=42., time=_now)
        ]

        # create non-influential mock environment influences, too
        _influences = [
            dei.Influence(
                material_color=dei.StringParameter(_material.material_color,
                                                   False),
                material_type=dei.StringParameter(_material.material_type,
                                                  False),
                material_producer=dei.StringParameter(
                    _material.material_producer, False),
                temperature=dei.NumericParameter(_measurements[0].temperature,
                                                 False),
                humidity=dei.NumericParameter(_measurements[0].humidity,
                                              False),
            )
        ]

        experiment_index = experiment_series.experiments.index(experiment)

        # define the selected parameters as a changed UI parameter
        if experiment_index != 0:
            _changed_ui_parameters = [
                dei.ChangedUIParameter(
                    label=parameter,
                    user_value=str(experiment.parameters[parameter]),
                    key=parameter
                )
                for parameter in experiment.adjusted_parameters
            ]
        else:
            _changed_ui_parameters = []

        lri: Optional[dei.LastRatingInfluence] = None
        if experiment_index != 0:
            # LastRatingInfluences excpects camelCase dict keys instead of snake_case.
            last_rating_influences_dict: LRIDictType = {
                to_camel_case(key): {
                    "value": value,
                    "influential": key in experiment.optimized_qualities
                }
                for key, value in experiment.qualities.items()
            }
            last_rating_influences_dict['experiment_id'] = {
                "value": experiment_series.experiments[experiment_index - 1].experiment_id,
                "influential": False
            }

            # rename overall_ok to comply to interface as it is expected in the AIPE database
            ok_tup = last_rating_influences_dict["overallOk"]
            del last_rating_influences_dict["overallOk"]
            last_rating_influences_dict["overall"] = ok_tup

            last_rating_influences_dict['comment'] = {
                "value": "created by SDG",
                "influential": False
            }
            lri = dei.LastRatingInfluence.from_dict(last_rating_influences_dict)

        _last_rating_influences = [lri]

        # bundle it all up into an Insight
        _insights = dei.Insights(
            comment="created by SDG",
            influences=_influences,
            changed_ui_parameters=_changed_ui_parameters,
            user_id=_user_id,
            uncertainty=1.0,
            last_rating_influences=_last_rating_influences)
        # finally, wrap it all up into a nice experiment 🎁
        parsed_experiment = dei.ExperimentFromDb(
            id=_experiment_id,
            printer="sdg_printer",
            stl_file_id=_stl_id,
            material=_material,
            oneoff=False,
            all_parameters=experiment.parameters,
            insights=_insights,
            measurements=_measurements,
            completion=1.0,
            ratings=[_rating],
            series_id=_series_id)

        return parsed_experiment

    @staticmethod
    def get_random_id() -> int:
        """ generate a random uuid fitted into an integer"""
        return uuid.uuid1().int >> 64
