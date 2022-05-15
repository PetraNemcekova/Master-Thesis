import numpy as np
import pandas as pd
import functions as fcn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# # Loading preprocessed data
met_preprocessed = pd.read_csv('metabolome1_preprocessed.csv', index_col=0).T
gen_preprocessed = pd.read_csv('SNP_gen1_preprocessed.csv', index_col=0).T

met_arr, gen_arr, gen1_feature_names,  identicifators = fcn.PrepareForPrediction(met_preprocessed, gen_preprocessed) 

# deviding into training and testing set

met_train, met_test, gen_train, gen_test = train_test_split(met_arr, gen_arr, test_size = 0.3,random_state=0)

# RF multivariate


RF = RandomForestRegressor(n_estimators = 80)
RF.fit(gen_train.values, met_train.values)
met_pred = RF.predict(gen_test.values)

# Evaluation
[max_mae, min_mae, mean_mae, sd_mae, max_mse, min_mse, mean_mse, sd_mse, max_cc, min_cc, mean_cc, sd_cc] = fcn.evaluateMultiModel(met_test.values, met_pred)

print(max_mae, min_mae, mean_mae, sd_mae)
print(max_mse, min_mse, mean_mse, sd_mse)
print(max_cc, min_cc, mean_cc, sd_cc) 

# RF univariate

metabolites_results =  np.zeros((met_train.shape[1], 3)) # 'Epsilons', 'C_param', 'MAE', 'MSE', 'CC'


for i in range(met_train.shape[1]):
    metRF = RandomForestRegressor()
    metRF.fit(gen_train.values, met_train.values[:,i])
    met_pred = metRF.predict(gen_test.values)
    [mae, mse, cc] = fcn.evaluationParametersSingleMetaboliteModel(met_test.values[:,i], met_pred)
    metabolites_results[i,0:3] = [mae, mse, cc]

# Evaluation

indexes = met_preprocessed.columns
metabolites_results_df = pd.DataFrame(metabolites_results, columns=[['MAE', 'MSE', 'PC']], index = indexes)

print(metabolites_results_df)
metabolites_results_df.to_csv('RF_03test_univariate_metabolites_results.csv')