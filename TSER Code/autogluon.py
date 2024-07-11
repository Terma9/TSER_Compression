import pandas as pd 
import numpy as np

import mlflow
from tsfeatures import tsfeatures


#Load already prepared data
#train_data = pd.read_csv('/home/sim/Desktop/TS Extrinsic Regression/Checking Out Data/data/prepared_data/AppliancesEnergy_TRAIN_ts_and_features.csv')
train_data = pd.read_csv('/home/sim/Desktop/TS Extrinsic Regression/data/prepared_data/Covid3Month_TRAIN_features.csv')

#test_data = pd.read_csv('/home/sim/Desktop/TS Extrinsic Regression/Checking Out Data/data/prepared_data/AppliancesEnergy_TEST_ts_and_features.csv')
test_data = pd.read_csv('/home/sim/Desktop/TS Extrinsic Regression/data/prepared_data/Covid3Month_TEST_features.csv')

test_y = test_data["target"].values

mlflow.set_tracking_uri("http://127.0.0.1:5001")

#mlflow.set_experiment("BeijingPM25_Dataset")
mlflow.set_experiment("ApplianacesEnergy_Dataset")
#mlflow.set_experiment("IEEEPPG_Dataset")

#Autogluon Training
from autogluon.tabular import TabularDataset, TabularPredictor

label = 'target'
#print(train_data[label].describe())

agluon_settings = {
    "time_limit" : 120,
    "presets" : "good_quality"
}

# Autogluon needs train_data and target in one df, target has to be named target
predictor = TabularPredictor(label=label).fit(train_data, **agluon_settings)

#Autogluon Prediction
predictions = predictor.predict(test_data.drop(columns=["target"]))

#print(predictor.evaluate(test_data, silent=True))
#print(predictor.leaderboard(test_data))


#See how well the model performed
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(test_y, predictions)
rmse = np.sqrt(mse)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(test_y, predictions)

# Calculate R-squared score
r_squared = r2_score(test_y, predictions)

metrics = {
    "RMSE" : rmse,
    "MAE" : mae,
    "R2" : r_squared
}

print("RMSE:", rmse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r_squared)


# Log with ml flow
run_name = "Autogluon 5min f (z-normalisation) (run2#)"


with mlflow.start_run(run_name=run_name) as run:


    mlflow.sklearn.log_model(predictor, 'agluon_predictor')

    # Log the automl params
    mlflow.log_params(agluon_settings)

    # Log fwiz features
    mlflow.log_metrics({
        "Started with Features" : train_data.shape[1],
        })

    # Log the error metrics that were calculated during validation
    mlflow.log_metrics(metrics)


