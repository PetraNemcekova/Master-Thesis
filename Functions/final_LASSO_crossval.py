import numpy as np
import pandas as pd
import functions as fcn
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import MultiTaskLasso, MultiTaskLassoCV, LassoCV

# Loading preprocessed data
met_preprocessed = pd.read_csv('metabolome1_preprocessed.csv', index_col=0).T
gen_preprocessed = pd.read_csv('SNP_gen1_preprocessed.csv', index_col=0).T

met_arr, gen_arr, gen1_feature_names, identicifators = fcn.PrepareForPrediction(met_preprocessed, gen_preprocessed) 

# LASSO

lasso_model = MultiTaskLassoCV(cv=5)
lasso_model.fit(gen_arr.values, met_arr.values)
best_alpha = lasso_model.alpha_

lasso_best = MultiTaskLasso(alpha=lasso_model.alpha_)
met_pred = cross_val_predict(lasso_best, gen_arr, met_arr, cv=5)

# Evaluation

[max_mae, min_mae, mean_mae, sd_mae, max_mse, min_mse, mean_mse, sd_mse, max_cc, min_cc, mean_cc, sd_cc] = fcn.evaluateMultiModel(met_arr.values, met_pred)

print(max_mae, min_mae, mean_mae, sd_mae)
print(max_mse, min_mse, mean_mse, sd_mse)
print(max_cc, min_cc, mean_cc, sd_cc) 
# Univariate

metabolites_results =  np.zeros((met_arr.shape[1], 3)) # 'Epsilons', 'C_param', 'MAE', 'MSE', 'CC'

for i in range(met_arr.shape[1]):

    lasso_model = LassoCV(cv=5)
    lasso_model.fit(gen_arr.values, met_arr.values[:,i])
    met_pred = cross_val_predict(lasso_model, gen_arr.values,met_arr.values[:,i], cv=5)
    [mae, mse, cc] = fcn.evaluationParametersSingleMetaboliteModel(met_arr.values[:,i], met_pred)
    metabolites_results[i,0:3] = [mae, mse, cc]

# Evaluation

indexes = met_preprocessed.columns
metabolites_results_df = pd.DataFrame(metabolites_results, columns=[['MAE', 'MSE', 'PC']], index = indexes)

print(metabolites_results_df)
metabolites_results_df.to_csv('LASSO_crossval_univariate_metabolites_results.csv')
