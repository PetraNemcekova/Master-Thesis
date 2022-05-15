import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict

import functions as fcn

# Loading preprocessed data
met_preprocessed = pd.read_csv('metabolome1_preprocessed.csv', index_col=0).T
gen_preprocessed = pd.read_csv('SNP_gen1_preprocessed.csv', index_col=0).T

met_arr, gen_arr, gen1_feature_names, identificators = fcn.PrepareForPrediction(
    met_preprocessed, gen_preprocessed)

# PLS

myPLS = PLSRegression()
met_pred = cross_val_predict(myPLS, gen_arr.values, met_arr.values, cv=5)

# Evaluation
[max_mae, min_mae, mean_mae, sd_mae, max_mse, min_mse, mean_mse, sd_mse, max_cc,
    min_cc, mean_cc, sd_cc] = fcn.evaluateMultiModel(met_arr.values, met_pred)

print(max_mae, min_mae, mean_mae, sd_mae)
print(max_mse, min_mse, mean_mse, sd_mse)
print(max_cc, min_cc, mean_cc, sd_cc)

# Univariate

# 'Epsilons', 'C_param', 'MAE', 'MSE', 'CC'
metabolites_results = np.zeros((met_arr.shape[1], 3))

for i in range(met_arr.shape[1]):
    met = met_arr.values[:, i]
    myPLS = PLSRegression()
    met_pred = cross_val_predict(
        myPLS, gen_arr.values, met.reshape(-1, 1), cv=5)
    [mae, mse, cc] = fcn.evaluationParametersSingleMetaboliteModel(
        met_arr.values[:, i], met_pred[:, 0])
    metabolites_results[i, 0:3] = [mae, mse, cc]

# Evaluation

indexes = met_preprocessed.columns
metabolites_results_df = pd.DataFrame(metabolites_results, columns=[
                                      ['MAE', 'MSE', 'PC']], index=indexes)

print(metabolites_results_df)
metabolites_results_df.to_csv(
    'sPLS_crossval_univariate_metabolites_results.csv')
