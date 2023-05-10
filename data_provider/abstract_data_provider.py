from __future__ import annotations

import distutils.util
import json
import re
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd
import igraph

import knowledge_graph
from utilities import camel_to_snake_ratings
from .synthetic_data_generation.database_experiment_interfaces import ExperimentSeries, ExperimentDict
from utils.preprocessing import LabelEncoderForColumns


class AbstractDataProvider(metaclass=ABCMeta):
    # keywords that should be ignored in the aggregate method
    _default_unused_keys = ['user_id', 'rating_date', 'time', 'comment']
    # the default attribute set to fetch from the MongoDB
    _default_experiment_attributes_to_fetch = [
        '_id', 'ratings', 'printer', 'changed_parameters', 'all_parameters',
        'oneoff', 'material', 'measurements', 'completion'
    ]

    def __init__(self):
        self._edges_dict = dict()

    def get_edges_dict(self):
        return self._edges_dict

    @classmethod
    def aggregate(cls,
                  value_dicts,
                  aggregate_function=np.mean,
                  value_names_to_ignore=None):
        """
        method to aggregate ratings
        :param value_dicts: a collection of dictionaries storing the values to aggregate
        :param aggregate_function: the function to aggregate all ratings. Defaults to np.mean.
        This can be a function accepting an array as well as 'mean' for np.mean or 'median' for np.median
        :param value_names_to_ignore: list of value_names that should be ignored and not aggregated
        :return: a dictionary with all aggregated values
        """
        if value_names_to_ignore is None:
            # if there is no specification of the names to ignore the _unused_names attribute as default
            value_names_to_ignore = cls._default_unused_keys
        # the defaultdict(list) constructor provides a dict that creates lists as default values for new entries
        aggregations = defaultdict(list)
        for value in value_dicts:
            for key in value:
                # if the name is is value_names_to_ignore the name should be discarded in the aggregation
                if key not in value_names_to_ignore:
                    aggregations[key].append(value[key])
        if aggregate_function == 'mean':
            avg_func = np.mean
        elif aggregate_function == 'median':
            avg_func = np.median
        else:
            avg_func = aggregate_function
        # the returned dictionary contains the keys of the input dictionaries the aggregation calculated from the
        # specified avg_func
        return {key: avg_func(values) for key, values in aggregations.items()}

    def _filter_duplicate_ratings(self, experiment):
        pass

    def _parse_experiment(self, experiment, process_parameters,
                          rating_aggregation_func, measure_aggregation_func):
        experiments_parameters = {
            'ID': experiment['_id'],
            'completion': experiment['completion'],
            'printer': experiment['printer']
        }
        # Builds up an dictionary with data of Database
        if 'changed_parameters' in experiment:
            changed_parameters = experiment['changed_parameters']
        else:
            changed_parameters = []
        unfloatable_parameternames = defaultdict(set)
        boolean_parameternames = set()
        for parameter_name in process_parameters:
            try:
                if isinstance(experiment['all_parameters'][parameter_name],
                              bool):
                    boolean_parameternames.add(parameter_name)
                experiments_parameters[parameter_name] = changed_parameters[
                    parameter_name] if parameter_name in changed_parameters \
                    else float(experiment['all_parameters'][parameter_name])
            except ValueError:
                # try converting string to bool, if it doesn't succeed treat it as a unfloatable value
                try:
                    experiments_parameters[parameter_name] = changed_parameters[
                        parameter_name] if parameter_name in changed_parameters else float(
                            distutils.util.strtobool(
                                experiment['all_parameters'][parameter_name]))
                    boolean_parameternames.add(parameter_name)
                except ValueError:
                    # if we encounter a value that is not floatable we keep the string representation and add the parametername & value to a dictionary that is used for applying label encoding later on
                    experiments_parameters[parameter_name] = changed_parameters[
                        parameter_name] if parameter_name in changed_parameters \
                        else experiment['all_parameters'][parameter_name]
                    unfloatable_parameternames[parameter_name] = {
                        experiments_parameters[parameter_name]
                    }
            except KeyError:
                # there might not be all keys present
                pass

        self._filter_duplicate_ratings(experiment)
        # append complex types
        experiments_parameters.update(
            self.aggregate(experiment['ratings'], rating_aggregation_func))
        experiments_parameters.update(
            self.aggregate(experiment['measurements'],
                           measure_aggregation_func))
        experiments_parameters.update(experiment['material'])
        return experiments_parameters, (unfloatable_parameternames,
                                        boolean_parameternames)

    def _fix_not_labelable(self, exp_list):
        for e in exp_list:
            for r in e['ratings']:
                if r['not_labelable']:
                    r['warping'] = 5

    @abstractmethod
    def _prepare_experiment_data(
            self, completed_only, containing_insights_only,
            experiment_series_id, include_oneoffs, labelable_only, limit,
            material_type,
            printer) -> tuple[ExperimentSeries, list[ExperimentDict]]:
        pass

    def get_experiments_with_graphs(self,
                                    influential_influences_only: bool,
                                    include_oneoffs=True,
                                    containing_insights_only=True,
                                    labelable_only=False,
                                    limit: int | None = None,
                                    label_encoder: LabelEncoderForColumns
                                    | None = None) -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, set[Any]], set[str], list[igraph.Graph], ExperimentSeries, LabelEncoderForColumns]:
        """returns the same as `get_executed_experiments_data`, but additionally parses the experiments/their influences to knowledge graphs

        Args:
            influential_influences_only (bool): if only quality influences marked as `influential` should be considered
            include_oneoffs (bool, optional): whether to include oneoffs into the data, True includes oneoffs, False excludes them, 'only' excludes non-oneoffs. Defaults to True.
            containing_insights_only (bool, optional): _description_. Defaults to True.
            labelable_only (bool, optional): if this is True only experiments rated as `labelable` are retrieved. Defaults to False.
            limit (int | None, optional): limit the number of returned objects from the data source. Defaults to None.
            label_encoder (LabelEncoderForColumns | None, optional): LabelEncoder used for experiment column en-/decoding. Defaults to None.

        Returns:
            tuple[pd.DataFrame, dict[str, set[str]], dict[str, set[Any]], set[str], list[igraph.Graph], ExperimentSeries, LabelEncoderForColumns]: 
            - label encoded dataframe of parsed experiments
            - dict of process-parameters & ratings
            - dict of other parameters (printer, material)
            - set of boolean parameter names
            - list of Knowledge Graphs created from insights of each experiment
            - experiment series mapping
            - label encoder

        """

        df, lok, lov, boolean_parameters, experiments, experiment_series = self.get_executed_experiments_data(
            include_oneoffs=include_oneoffs,
            completed_only=False,
            labelable_only=labelable_only,
            containing_insights_only=containing_insights_only,
            limit=limit)
        df = df.sort_index(axis=1)

        if label_encoder is None:
            from utils.preprocessing import Preprocessing
            label_encoded_df, column_transformer = Preprocessing.label_encoding(
                df, [k[:-1] for k in lov.keys()])
            assert column_transformer is not None
            label_encoder = LabelEncoderForColumns(
                column_transformer.transformers_[0][1],
                [k[:-1] for k in lov.keys()], column_transformer)
        else:
            label_encoded_df = label_encoder.transformdf(
                df, [k[:-1] for k in lov.keys()])

        parsed_influence_graphs = []
        for experiment in experiments:
            try:
                # only consider experiments with insights for knowledge graph creation
                if 'insights' in experiment.keys():
                    parsed_influence_graphs.append(
                        knowledge_graph.parse_insights_to_graph(
                            experiment,
                            label_encoder,
                            influential_only=influential_influences_only,
                            edges_dict=self.get_edges_dict()))
            except Exception as e:
                print(f'Exception of {type(e)} {e} in id {experiment["_id"]}')
                continue
        return label_encoded_df, lok, lov, boolean_parameters, parsed_influence_graphs, experiment_series, label_encoder

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
               set[str], list[ExperimentDict], ExperimentSeries]:
        """Returns experiment data set from the respective data source
        NOTICE: most filter parameters are only applied in the AipeDataProvider implementation! 

        Args:
            rating_averaging (str | Callable, optional): averaging function for ratings. accepts 'mean', 'median' or callable. Defaults to 'mean'.
            measurement_averaging (str | Callable, optional): averaging function for temperature and humidity measurements. accepts 'mean', 'median' or callable. Defaults to 'median'.
            include_oneoffs (str | bool, optional): whether to include oneoffs into the data, True includes oneoffs, False excludes them, 'only' excludes non-oneoffs. Defaults to False.
            printer (str | None, optional): request data only for a specific printer. If None, all found printers will be used. Defaults to None.
            experiment_series_id (str | None, optional): request data only for a specific experiment_series (object_id as string). If None, all found experiments will be used. Only possible if oneoffs are excluded. Defaults to None.
            completed_only (bool, optional): only retrieve those experiments that completed successfully. note that when this is False, `not_lableable` flag needs to be checked. Defaults to False.
            labelable_only (bool, optional): if this is True only experiments rated as `labelable` are retrieved. Defaults to True.
            containing_insights_only (bool, optional): _description_. Defaults to False.
            material_type (str | None, optional): queries for a specific material_type e.g. PETG. Defaults to None.
            limit (int | None, optional): limit the number of returned objects from the data source. Defaults to None.

        Returns:
            tuple[pd.DataFrame, dict[str, set[str]], dict[str, set[Any]], set[str], list[ ExperimentDict], ExperimentSeries]: 
            - dataframe of parsed experiments
            - dict of process-parameters & ratings
            - dict of other parameters (printer, material)
            - set of boolean parameter names
            - list of experiments (as dicts)
            - experiment series mapping
        """
        filters = list()
        if printer is not None:
            filters.append(lambda x: printer == x['printer'])

        experiment_series, experiments = self._prepare_experiment_data(
            completed_only, containing_insights_only, experiment_series_id,
            include_oneoffs, labelable_only, limit, material_type, printer)
        # prevents duplications of parameter-names
        processparams = set()
        # prevents duplications of ratings
        # also enables set-specific methods
        ratings = set()
        material_producers = set()
        material_types = set()
        material_colors = set()
        printers = set()
        # if we only pull oneoffs we still want to include these parameters
        #  for most series they will automatically included below
        if include_oneoffs == 'only':
            processparams |= {
                'retraction_speed', 'speed_print', 'speed_travel',
                'retraction_amount', 'material_bed_temperature',
                'cool_fan_speed', 'material_print_temperature'
            }
        for experiment in experiments:
            if 'changed_parameters' in experiment:
                processparams |= set(experiment['changed_parameters'].keys())

            if 'insights' in experiment.keys():
                if 'changed_ui_parameters' in experiment['insights'].keys():
                    # iterate through all process parameter changes made in cura
                    # we need to iterate backwards otherwise we might miss some parameters that are the same as the default
                    for parameter in reversed(
                            experiment['insights']['changed_ui_parameters']):
                        # cura does not always detect if a process parameter has the default value if it was changed before
                        # so we have to remove the process parameters that have the same value as the default one
                        if parameter['original_value'] == parameter[
                                'user_value']:
                            experiment['insights'][
                                'changed_ui_parameters'].remove(parameter)
                processparams |= set([
                    e['key']
                    for e in experiment['insights']['changed_ui_parameters']
                ])
                # rename influences
                if 'last_rating_influences' in experiment['insights'].keys():
                    renamed_influences = {}
                    # we currently support only one previous experiment
                    influences = experiment['insights'][
                        'last_rating_influences'][0]
                    if influences is not None:
                        # should only be one element but better safe than sorry
                        for influence_key in influences.keys():
                            renamed_key = camel_to_snake_ratings(influence_key)
                            renamed_influences.update(
                                {renamed_key: influences[influence_key]})
                        experiment['insights']['last_rating_influences'] = [
                            renamed_influences
                        ]
                    # else:
                    #     # weird edge case which causes an index error for empty lists
                    #     experiment['insights']['last_rating_influences'] = [None]

            ratings |= set(experiment['ratings'][0].keys())
            # don't despair - this notation is simply there to make sets accept strings - you could also add them as an array...
            # making all producers uppercase so we don't get the same producer two times, e.g. Kaisertech and KaiserTech
            material_producers |= {
                experiment['material']['material_producer'].upper()
            }
            material_types |= {experiment['material']['material_type'].upper()}
            material_colors |= {experiment['material']['material_color']}
            printers |= {experiment['printer']}
        # remove this string-set from the rating-set
        ratings -= {'user_id', 'rating_date', 'comment'}
        parsed_experiments = []
        unparsable_lov = defaultdict(set)
        boolean_parameters = set()
        for experiment in experiments:
            if filters is not None and not all(
                    [f(experiment) for f in filters]):
                # there exist a filter that evaluates to false
                continue
            try:
                parsed_experiment, nonfloat_params = self._parse_experiment(
                    experiment,
                    process_parameters=processparams,
                    rating_aggregation_func=rating_averaging,
                    measure_aggregation_func=measurement_averaging)
                parsed_experiments.append(parsed_experiment)
                unfloatable_params, boolean_params = nonfloat_params
                boolean_parameters = boolean_params | boolean_parameters
                for param, value in unfloatable_params.items():
                    unparsable_lov[param] = unparsable_lov[param] | value

            except Exception as e:
                print(f'Exception of {type(e)} {e} in id {experiment["_id"]}')
                continue

        with open('./obj/sdg_boolean_parameters.txt', 'w', encoding='utf8') as f:
            f.write(json.dumps(sorted(list(boolean_parameters))))
        # Creates lists and Dictionary and returns them
        # TODO include a list of all other keys, like completion, printer etc.
        # TODO split ratings into binary and categorical

        df = pd.DataFrame.from_dict(parsed_experiments)

        # Remove all whitespaces from a String and change it to lowercase
        def normalize_strings(x):
            return x.lower().replace(" ", "")

        # Apply mapping on df entries and on material_colors, material_types and material_producers set
        if 'material_color' in df.columns:
            df['material_color'] = df['material_color'].map(normalize_strings)
            material_colors = set(
                [normalize_strings(x) for x in material_colors])
        if 'material_type' in df.columns:
            df['material_type'] = df['material_type'].map(normalize_strings)
            material_types = set(
                [normalize_strings(x) for x in material_types])
        if 'material_producer' in df.columns:
            df['material_producer'] = df['material_producer'].map(
                normalize_strings)
            material_producers = set(
                [normalize_strings(x) for x in material_producers])

        print("Sets of Data (Experiments): {0}\n".format(len(df)))

        lov = {
            'printers': printers,
            'material_colors': material_colors,
            'material_types': material_types,
            'material_producers': material_producers
        }
        for unfloatable_param_key, values in unparsable_lov.items():
            lov.update({unfloatable_param_key + 's': values})

        return (
            df,
            {
                'process_parameters': processparams,  # parameter keys
                'ratings': ratings  # rating keys
            },
            lov,
            boolean_parameters,
            experiments,
            experiment_series)

    def get_connected_experiments_by_last_influences(self,
                                                     include_oneoffs=False,
                                                     printer=None,
                                                     experiment_series_id=None,
                                                     completed_only=False,
                                                     labelable_only=True,
                                                     containing_insights_only=False,
                                                     material_type=None,
                                                     limit=None):
        """
        Returns list of tuples of experiment_ids that are connected through a last_rating reference
        :param experiment_series_id: request data only for a specific experiment_series (object_id as string).
                        If None, all found experiments will be used
                        Only possible if oneoffs are excluded
        :param include_oneoffs: whether to include oneoffs into the data,
        True includes oneoffs, False excludes them, 'only' excludes non-oneoffs
        :param printer: request data only for a specific printer (name as string).
                        If None, all found printers will be used
        :param completed_only: only retrieve those experiments that completed successfully. note that when this
        is False
                        not_lableable flag needs to be checked
        :param labelable_only: if this is True only experiments rated as labelable are retrieved
        :param material_type queries for a specific material_type e.g. PETG (string)
        :param limit: limit the number of returned objects from the database
        :return: list of 2-tuples that hold the experiment_ids from connected experiments
        """
        _, experiments = self._prepare_experiment_data(
            completed_only, containing_insights_only, experiment_series_id,
            include_oneoffs, labelable_only, limit, material_type, printer)

        id_tuple_list = []
        # iterate experiment and check if experiments are connected through last_rating_influences
        for e in experiments:
            if 'last_rating_influences' in e['insights']:
                if e['insights']['last_rating_influences'] is not None:
                    if e['insights']['last_rating_influences'][0] is not None:
                        current_id: int = int(e['_id'])
                        previous_id: int = int(
                            e['insights']['last_rating_influences'][0]['experiment_id'])
                        id_tuple_list.append((previous_id, current_id))
        return id_tuple_list
