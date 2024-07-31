########### 
######## When everything runs nicely, add code to print sysout to an output file in the run directory
##### Also no logging of Datasets bc no space!


import os, socket


if socket.gethostname() != "sim-IdeaPad-5-14ALC05":
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = "40" 


import argparse, shutil, pickle
from datetime import datetime
import pandas as pd
import numpy as np

import mlflow
from featurewiz import FeatureWiz
from flaml import AutoML

from utils.personal_utils import *


# '/home/simon/TSER/Covid3Month_TRAIN_features.csv'


def run_flaml(source_path_train, experiment_name, run_name, time):



    # Get paths for the folders. I save one folder outside of cwd. Is at Beginning for Saving Output in run-folder.
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    log_folder_path = os.path.join(parent_directory, 'All-Logs')
    experiment_path = os.path.join(log_folder_path, experiment_name)
    run_path = os.path.join(experiment_path, run_name)
    
    # Check if the log-folder exists; if not, create it
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)
    
    # Check if the experiment folder exists; if not, create it
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    
    # Remove the existing run folder if it exists and create it
    if os.path.exists(run_path):
        shutil.rmtree(run_path)
    os.makedirs(run_path)



    # Ml-Flow Setup
    mlflow.set_tracking_uri("http://127.0.0.1:5003")
    mlflow.set_experiment(experiment_name=experiment_name)



    #Load already prepared data
    train_data = pd.read_csv(source_path_train)
    test_data = pd.read_csv(source_path_train.replace('TRAIN', 'TEST'))
    test_y = test_data["target"].values



    # Automated feature Selection 
    # no activated feature engg, takes too long, feature_engg='interactions'
    fwiz = FeatureWiz(transform_target=True, verbose=2,) # see other parameters on doc
    train_data_fw , train_y_fw  = fwiz.fit_transform(train_data.drop(columns=['target']), train_data["target"])
    # apply same learned feature transformations on test data _-> nothing with current params??
    test_data_fw = fwiz.transform(test_data.drop(columns=['target']))

    #Get list of selected features 
    selected_features = fwiz.features 


    #Selecting Model and Tuning Hyperparameters with FLAML
    flaml_log_name = f"flmal-log.log"

    automl_settings = {
        "task" : "regression",
        "metric" : "rmse",
        "time_budget" : time,
        "log_file_name" : flaml_log_name,
        ##"n_jobs" : 24
    }

    automl = AutoML()
    automl.fit(X_train= train_data_fw, y_train = train_data['target'], **automl_settings)
    predictions = automl.predict(test_data_fw)

    #See how well the model performed
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score

    mse = mean_squared_error(test_y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_y, predictions)
    r_squared = r2_score(test_y, predictions)

    # Calculate the MAPE Metrics
    mape = get_mape(test_y, predictions)
    smape = get_smape(test_y, predictions)
    msmape = get_msmape(test_y, predictions)

    mape_sk = mean_absolute_percentage_error(test_y, predictions)
    explVar = explained_variance_score(test_y, predictions)

    metrics = {
        "explVar": explVar,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r_squared,
        "MAPE": mape,
        "sMAPE": smape,
        "msMAPE": msmape,
        "MAPE sklearn": mape_sk,
        "Started with Features": train_data.shape[1],
        "Selected Features": train_data_fw.shape[1]
    }

    print(metrics)




    # Start Logging
    # Update param dictionary with new parameters
    automl_settings.pop('log_file_name')
    automl_settings["Source-Dataset"] = source_path_train
    automl_settings["Experiment"] = experiment_name
    automl_settings["Run"] = run_name

    time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    automl_settings["Time of run"] = time_string
    

    # Log with ml flow
    with mlflow.start_run(run_name=run_name) as run:
        
        #mlflow.sklearn.log_model(automl, 'automl')
        # Log the automl params
        mlflow.log_params(automl_settings)
        # Log the error metrics that were calculated during validation
        mlflow.log_metrics(metrics)



    # Path for the metrics and settings text file
    metrics_file_path = os.path.join(run_path, 'metrics_and_settings.txt')
    # Save the metrics and settings to the text file
    with open(metrics_file_path, 'w') as file:
        file.write("Metrics:\n")
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")
        
        file.write("\nAutoML Settings:\n")
        for key, value in automl_settings.items():
            file.write(f"{key}: {value}\n")


    # Move the flaml log to run directory -> can cause problems later, be cautionous!
    log_file_path = os.path.join(current_directory, flaml_log_name)
    # Copy the log file to the run folder
    if os.path.exists(log_file_path):
        shutil.copy(log_file_path, run_path)
    else:
        print(f"Log file {log_file_path} does not exist.")



    # Save Starting Datasets
    #train_data.to_csv(os.path.join(run_path, 'train_data.csv'), index=False)
    #test_data.to_csv(os.path.join(run_path, 'test_data.csv'), index=False)

    # Save Selected Features
    with open(os.path.join(run_path, 'selected_features.txt'), 'w') as file:
        for feature in selected_features:
            file.write(f"{feature}\n")

    # Save predictions as a NumPy binary file
    np.save(os.path.join(run_path, 'predictions.npy'), predictions)

    # Save the model as pkl
    model_path = os.path.join(run_path, 'flaml_model.pkl')
    with open(model_path, "wb") as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)




    return rmse.item()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLAML with specified parameters.")
    parser.add_argument("source_path_train", type=str, help="Path to the source CSV file")
    parser.add_argument("experiment_name", type=str, help="Name of the MLflow experiment")
    parser.add_argument("run_name", type=str, help="Name of the MLflow run")
    parser.add_argument("time", type=int, help="Time budget for FLAML in seconds")

    args = parser.parse_args()
    run_flaml(args.source_path_train, args.experiment_name, args.run_name, args.time)
