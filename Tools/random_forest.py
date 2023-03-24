# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def fit_and_report(model, x, y, description):
    model.fit(x, y)
    feature_names = model.feature_names_in_.tolist()
    importances = model.feature_importances_.tolist()
    importances = [round(val * 100, 2) for val in importances]

    print(f"Feature importances for {description}")
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance}")

in_file = sys.argv[1]
num_insertions = int(sys.argv[2])
independents_to_drop = sys.argv[3:] if len(sys.argv) > 2 else []
dataframe = pd.read_csv(in_file)

dataframe = dataframe[dataframe['num_insertions'] == num_insertions]
dataframe = dataframe.drop('num_insertions', axis=1)

x = dataframe.drop(independents_to_drop + ['canaries_greedy', 'canaries_beam', 'secret_nll'], axis=1)
y_greedy = dataframe['canaries_greedy']
y_beam = dataframe['canaries_beam']
y_nll = dataframe['secret_nll']

model = RandomForestRegressor(n_estimators=64, max_depth=8, random_state=1)

fit_and_report(model, x, y_greedy, "greedy extraction")
print("")
fit_and_report(model, x, y_beam, "beam-search extraction")
print("")
fit_and_report(model, x, y_nll, "secret negative log likelihood")
