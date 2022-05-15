import os
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error


def replaceColumnsWith1stRowValues(dataframe):

    result = dataframe
    # fix headers with first row (names was previously in first column)
    result.columns = result.iloc[0]
    # drop first row
    result = result.iloc[1:, :]
    return result


def replaceIndexWith1stColumnValues(dataframe):

    result = dataframe
    # fix index/rows with first column data
    result.index = result.iloc[:, 0].values
    firstColumnName = result.columns.values[0]
    # drop first column
    result = result.drop([firstColumnName], axis=1)
    return result

# import csv


def getFilePath(relativePath):
    return os.path.join(os.path.dirname(__file__), relativePath)


def getInterselectedWithFamilies(metabolome_dataframe, genome_dataframe):

    # fix headers with first row

    genome_families_list = genome_dataframe.columns.values.tolist()
    metabolome_families_list = metabolome_dataframe.columns.values.tolist()

    metabolome_families_intersected = np.intersect1d(
        metabolome_families_list, genome_families_list)

    genome_dataframe_intersected = genome_dataframe[metabolome_families_intersected]
    metabolome_dataframe_intersected = metabolome_dataframe[metabolome_families_intersected]

    return metabolome_dataframe_intersected, genome_dataframe_intersected


def deleteGenomeFeaturesWithNaN(data, features):
    data_preprocessed = data

    for i in range(len(features)):
        if (data.iloc[i, :].isna().sum() > 0):
            unused_features = features.iloc[i]
            data_preprocessed = data_preprocessed.drop(unused_features)

    return data_preprocessed


def replaceMetabolomeValuesWithNaN(data):
    data_preprocessed = data.fillna(0)
    return data_preprocessed


def standardizeValues(dataFrame: DataFrame) -> DataFrame:
    result = dataFrame.replace(2, -1)
    return result


def PrepareForPrediction(metabolomic, genomic):
    standardized1 = standardizeValues(genomic)

    # Shuffle

    perm = np.random.permutation(metabolomic.shape[0])
    names = metabolomic.iloc[perm, :].index

    metabolome1_mixed = metabolomic.iloc[perm, :]
    genome1_mixed = standardized1.iloc[perm, :]

    identicifators = metabolome1_mixed.index

    # To array

    # met1_arr = metabolome1_mixed.values
    # gen1_arr = genome1_mixed.values

    gen1_feature_names = list(genome1_mixed.columns)

    return metabolome1_mixed, genome1_mixed, gen1_feature_names, identicifators


def evaluateMultiModel(real_data, predicted_data):
    maes = []
    mses = []

    for i in range(real_data.shape[0]):
        mae = mean_absolute_error(real_data[i], predicted_data[i])
        maes.append(mae)
        mse = mean_squared_error(real_data[i], predicted_data[i])
        mses.append(mse)

    max_mae = np.max(maes)
    min_mae = np.min(maes)
    mean_mae = np.mean(maes)
    sd_mae = np.std(maes)

    max_mse = np.max(mses)
    min_mse = np.min(mses)
    mean_mse = np.mean(mses)
    sd_mse = np.std(mses)

    cc_matrix = np.corrcoef(real_data, predicted_data)[
        real_data.shape[0]:real_data.shape[0]*2, 0:real_data.shape[0]]
    cc_diagonal = np.diagonal(cc_matrix)
    max_cc = np.max(cc_diagonal)
    min_cc = np.min(cc_diagonal)
    mean_cc = np.mean(cc_diagonal)
    sd_cc = np.std(cc_diagonal)

    return max_mae, min_mae, mean_mae, sd_mae, max_mse, min_mse, mean_mse, sd_mse, max_cc, min_cc, mean_cc, sd_cc


def evaluationParametersSingleMetaboliteModel(real_data, predicted_data):

    mae = mean_absolute_error(real_data, predicted_data)
    mse = mean_squared_error(real_data, predicted_data)

    cc_matrix = np.corrcoef(real_data, predicted_data)
    cc = np.nan_to_num(cc_matrix[0, 1])

    return mae, mse, cc
