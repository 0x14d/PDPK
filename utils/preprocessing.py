from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
# We have to use OrdinalEncoder for label encoding
from sklearn.preprocessing import OrdinalEncoder


class LabelEncoderForColumns:
    """
    Utility functions to wrap an OrdinalEncoder trained for multiple columns
    """
    _label_encoder: OrdinalEncoder = None
    _columns: List[str] = []
    _column_transformer: ColumnTransformer = None

    def __init__(self, label_encoder: OrdinalEncoder, columns: List[str], column_transformer: ColumnTransformer = None):
        """

        :param label_encoder: Fitted OrdinalEncoder
        :param columns:
        """
        self._label_encoder = label_encoder
        self._columns = columns
        self._column_transformer = column_transformer

    def transform(self, str_to_transform: str, column_to_transform: str) -> int:
        """
        wraps the OrdinalEncoder transform. Constructing input array is necessary since we have one OrdinalEncoder that was trained on all columns -> we need to adjust the input accordingly
        :param str_to_transform:
        :param column_to_transform:
        :return:
        """
        index = self._columns.index(column_to_transform)
        input_array = [[cat[0] for cat in self._label_encoder.categories_]]
        input_array[0][index] = str_to_transform
        return self._label_encoder.transform(input_array)[0][index]

    def inverse_transform(self, int_to_transform:int, column_to_transform:str) -> str:
        """
        wraps the OrdinalEncoders' inverse_transform int -> str
        :param int_to_transform:
        :param column_to_transofrm:
        :return:
        """
        index = self._columns.index(column_to_transform)
        input_array = [[0]*len(self._columns)]
        input_array[0][index] = int_to_transform
        return self._label_encoder.inverse_transform(input_array)[0][index]

    def prepare_array(self, column_to_transform: str, value: Union[int,str]) -> Tuple[np.array, int]:
        index = self._columns.index(column_to_transform)
        input_array = [cat[0] for cat in self._label_encoder.categories_]
        input_array[0][index] = value
        return input_array, index

    def transformdf(self, X: pd.DataFrame, columns: List):
        _X = self._column_transformer.transform(X)
        update_dataframe = pd.DataFrame(_X, columns=columns)
        X.update(update_dataframe)
        X[columns] = X[columns].astype(np.int64)
        return X
