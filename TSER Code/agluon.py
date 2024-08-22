import os, socket

if socket.gethostname() != "sim-IdeaPad-5-14ALC05":
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = "40" 


import argparse, shutil, pickle
from datetime import datetime
import pandas as pd
import numpy as np

import mlflow
from utils.personal_utils import *

from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def run_agluon(experiment_name, run_name, time, train_data, test_data):



    # Get paths for the folders. I save one folder outside of cwd. Is at Beginning for Saving Output in run-folder.
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    log_folder_path = os.path.join(parent_directory, 'All-Logs_agluon')
    experiment_path = os.path.join(log_folder_path, experiment_name)
    run_path = os.path.join(experiment_path, run_name)
    model_folder_path = os.path.join(run_path, 'ModelFolder')
    
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


    # Remove the existing model folder if it exists and create it
    if os.path.exists(model_folder_path):
        shutil.rmtree(model_folder_path)
    os.makedirs(model_folder_path)




    test_y = test_data["target"].values

    mlflow.set_tracking_uri("http://127.0.0.1:5003")
    mlflow.set_experiment(experiment_name=experiment_name)


    #Autogluon Training
    
    #print(train_data[label].describe())

    automl_settings = {
        "time_limit" : time,
        "presets" : "good_quality", # best quality improves gives best result, but needs most resources!
        ##"num_cpus" : 24
    }


    logging_file_path = run_path + '/thelogs.txt'
    os.mknod(logging_file_path)



    # Autogluon needs train_data and target in one df, target has to be named target
    # !! to log normally change path to path= model_folder_path
    predictor = TabularPredictor(label='target', eval_metric='root_mean_squared_error', path='model_folder_path' , log_to_file=True, verbosity=3, log_file_path=logging_file_path).fit(train_data, **automl_settings)

    #Autogluon Prediction
    predictions = predictor.predict(test_data.drop(columns=["target"]))

    print(predictor.evaluate(test_data, silent=True))
    print(predictor.leaderboard(test_data))


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
    }

    print(metrics)


    automl_settings["Experiment"] = experiment_name
    automl_settings["Run"] = run_name

    time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    automl_settings["Time of run"] = time_string

    with mlflow.start_run(run_name=run_name) as run:

        #mlflow.sklearn.log_model(predictor, 'agluon_predictor')
        mlflow.log_params(automl_settings)

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


    leaderboard_df = predictor.leaderboard(test_data)
    # Specify the path to save the CSV or Excel file
    csv_path = os.path.join(run_path, 'leaderboard')
    # Save the DataFrame to a CSV file
    leaderboard_df.to_csv(csv_path, index=False)







    # Save predictions as a NumPy binary file
    np.save(os.path.join(run_path, 'predictions.npy'), predictions)



    # Agluon automatically saves model in defined path from TabularPredictor, later load simply with load





    return predictions
