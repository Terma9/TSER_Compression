
# How to use:
# To activate Fwiz: uncomment big code, add feature-after in metrics, use in flaml commented lines!
import pandas as pd
import numpy as np
import mlflow
from tsfeatures import tsfeatures

from featurewiz import FeatureWiz

from flaml import AutoML

#Load already prepared data
#train_data = pd.read_csv('/home/sim/Desktop/TS Extrinsic Regression/Checking Out Data/data/prepared_data/AppliancesEnergy_TRAIN_ts_and_features.csv')
train_data = pd.read_csv('/home/sim/Desktop/TS Extrinsic Regression/data/prepared_data/Covid3Month_TRAIN_features.csv')

#test_data = pd.read_csv('/home/sim/Desktop/TS Extrinsic Regression/Checking Out Data/data/prepared_data/AppliancesEnergy_TEST_ts_and_features.csv')
test_data = pd.read_csv('/home/sim/Desktop/TS Extrinsic Regression/data/prepared_data/Covid3Month_TEST_features.csv')

test_y = test_data["target"].values


mlflow.set_tracking_uri("http://127.0.0.1:5000")

#mlflow.set_experiment("BeijingPM25_Dataset")
mlflow.set_experiment("ApplianacesEnergy_Dataset")
#mlflow.set_experiment("IEEEPPG_Dataset")



 # Automated feature Selection 


# no activated feature engg, takes too long, feature_engg='interactions'
fwiz = FeatureWiz(transform_target=True, verbose=0, ) # see other parameters on doc

train_data_fw , train_y_fw  = fwiz.fit_transform(train_data.drop(columns=['target']), train_data["target"])

# apply same learned feature transformations on test data _-> nothing with current params??
test_data_fw = fwiz.transform(test_data.drop(columns=['target']))


### get list of selected features ###
fwiz.features 



# AutoML Pipeline

#Selecting Model and Tuning Hyperparameters with FLAML



# rsme doesn't exist
automl_settings = {
    "task" : "regression",
    "metric" : "mse",
    "time_budget" :100,
    "log_file_name" : "flaml_log_basic.log",

}

automl = AutoML()
# for only flaml
#automl.fit(X_train= train_data.drop(columns=['target']), y_train = train_data['target'], **automl_settings)
automl.fit(X_train= train_data_fw, y_train = train_data['target'], **automl_settings)

#See how well the model performed
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#for only flaml
#predictions = automl.predict(test_data.drop(columns=['target']))
predictions = automl.predict(test_data_fw)

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


#Ml-Flow Run Name
run_name = "FLAML and fwiz 5min f "


# Log with ml flow
with mlflow.start_run(run_name=run_name) as run:


    mlflow.sklearn.log_model(automl, 'automl')

    # Log the automl params
    mlflow.log_params(automl_settings)

##"Selected Features" : train_x_flattend_fw.shape[1]
    # Log fwiz features
    mlflow.log_metrics({
        "Started with Features" : train_data.shape[1],
        "Selected Features" : train_data_fw.shape[1]
        })

    # Log the error metrics that were calculated during validation
    mlflow.log_metrics(metrics)




