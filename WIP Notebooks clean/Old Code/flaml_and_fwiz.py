import os, socket

if socket.gethostname() != "sim-IdeaPad-5-14ALC05":
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = "40" 

# How to use:
# To activate Fwiz: uncomment big code, add feature-after in metrics, use in flaml commented lines!

import argparse
import pandas as pd
import numpy as np
import mlflow
from tsfeatures import tsfeatures

from featurewiz import FeatureWiz

from flaml import AutoML

from personal_utils import *

# '/home/simon/TSER/Covid3Month_TRAIN_features.csv'



def run_flaml(source_path_train, experiment_name, run_name, time):


    #Load already prepared data
    #train_data = pd.read_csv('/home/sim/Desktop/TS Extrinsic Regression/Checking Out Data/data/prepared_data/AppliancesEnergy_TRAIN_ts_and_features.csv')
    train_data = pd.read_csv(source_path_train)

    #test_data = pd.read_csv('/home/sim/Desktop/TS Extrinsic Regression/Checking Out Data/data/prepared_data/AppliancesEnergy_TEST_ts_and_features.csv')
    test_data = pd.read_csv(source_path_train.replace('TRAIN', 'TEST'))

    test_y = test_data["target"].values


    mlflow.set_tracking_uri("http://127.0.0.1:5001")

    #mlflow.set_experiment("BeijingPM25_Dataset")
    mlflow.set_experiment(experiment_name=experiment_name)
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



    automl_settings = {
        "task" : "regression",
        "metric" : "rmse",
        "time_budget" : time,
        "log_file_name" : "flaml_log_basic.log",
        ##"n_jobs" : 24


    }

    automl = AutoML()
    # for only flaml
    #automl.fit(X_train= train_data.drop(columns=['target']), y_train = train_data['target'], **automl_settings)
    automl.fit(X_train= train_data_fw, y_train = train_data['target'], **automl_settings)




    #See how well the model performed
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score

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


    # Calculate the MAPE Metrics

    mape = mape(test_y, predictions)
    smape = smape(test_y, predictions)
    msmape = msmape(test_y, predictions)
    mape_sk = mean_absolute_percentage_error(test_y, predictions)

    explVar = explained_variance_score(test_y, predictions)




    metrics = {
        "explVar": explVar,
        "RMSE" : rmse,
        "MAE" : mae,
        "R2" : r_squared,
        "MAPE": mape,
        "sMAPE": smape,
        "smMAPE": msmape,
        "MAPE sklearn": mape_sk

    }

    print("RMSE:", rmse)
    print("Mean Absolute Error:", mae)
    print("R-squared Score:", r_squared)
    print(metrics)


    #Ml-Flow Run Nam

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


    # item() because it is a NDArray[Any]
    return rmse.item()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLAML with specified parameters.")
    parser.add_argument("source_path_train", type=str, help="Path to the source CSV file")
    parser.add_argument("experiment_name", type=str, help="Name of the MLflow experiment")
    parser.add_argument("run_name", type=str, help="Name of the MLflow run")
    parser.add_argument("time", type=int, help="Time budget for FLAML in seconds")

    args = parser.parse_args()
    run_flaml(args.source_path_train, args.experiment_name, args.run_name, args.time)
