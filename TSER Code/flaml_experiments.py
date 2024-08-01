import os, socket

if socket.gethostname() != "sim-IdeaPad-5-14ALC05":
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = "40" 




import argparse
from flaml_and_fwiztest_wlogs import run_flaml
from utils.personal_utils import *

path = '/home/simon/TSER/preparedData/'

path_ts = '/home/simon/TSER/Time_Series_Data/'


# For Agluon change all occurences
# For new Dataset, change ds_name
# Maybe add parameter option to do all test at once! -> delete before that the old experiments!

def tsf_or_f():

    ds_names = [
    'AppliancesEnergy',
    'NewsTitleSentiment',
    'BenzeneConcentration',
    'BeijingPM25Quality',
    'IEEEPPG',
    'FloodModeling1',
    'HouseholdPowerConsumption1',
    'Covid3Month'
    ]


    for ds_name in ds_names:

        for time in [15]:
            source_path_tsf = path + ds_name + '_TRAIN_None__ts_and_features.csv'
            source_path_f = path + ds_name + '_TRAIN_None__features.csv'

            run_flaml(source_path_tsf, 'Test if tsf or f', f'tsf {ds_name} Flaml {time}min', time * 60)
            run_flaml(source_path_f, 'Test if tsf or f', f'f {ds_name} Flaml {time}min', time * 60)


def best_time():
    # Letting run only with features

    ds_names = [
    'AppliancesEnergy',
    'NewsTitleSentiment',
    'BenzeneConcentration',
    'BeijingPM25Quality',
    'IEEEPPG',
    'FloodModeling1',
    'HouseholdPowerConsumption1',
    'Covid3Month'
    ]

    times = [1 ,5, 15, 20, 25, 30, 60]

    for ds_name in ds_names:

        source_path = path + ds_name + '_TRAIN_None__features.csv'


        for time in times:
            run_flaml(source_path, 'Test_best_runtime', f'{time}min_{ds_name}_Flaml_f', time * 60)




def test_deterministic():
    source_path = path + ds_name + '_TRAIN_None__features.csv'

    for i in range(5):
        run_flaml(source_path, 'Test deterministic', f'Run{i+1} Appliances Flaml 5min', 5 * 60)




# I assume only f, 15min -> maybe change later


# Do the runs, put on mlflow
# Then calculte comp_ratios and save it as array -> transfer back, from that I can do bar chart!


# Workaround to create charts: return the two y values, from that i can calculate all the metrics
# Calculate comp_ratio by yourself!


def run_dataset():




    # First run with no compression. I take Compression Ratio one!
    ds_names = ['AppliancesEnergy','HouseholdPowerConsumption1', 'FloodModeling1',]



    # Create folder to save the outputs
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    folder_path = os.path.join(parent_directory, 'Output_Runs')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)



    for ds_name in ds_names:

        #!! I add each solution and overwrite in case
        ds_path = os.path.join(folder_path, ds_name)
        if not os.path.exists(ds_path):
            os.makedirs(ds_path)

        

        source_path = path + ds_name + '_TRAIN_None__features.csv'
        y_and_prediction_none = run_flaml(source_path, f'{ds_name} Runs ', f'NONE {ds_name} 15min Flaml f', 15 * 60)


    

        for tq in ['dct', 'dft', 'dwt']:

            values = []
            values.append(y_and_prediction_none)

            for i in [0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.99]:
                source_path = path + ds_name + '_TRAIN' + '_' + tq + f'_{i}' + '_features.csv'

                y_and_prediction = run_flaml(source_path, f'{ds_name} Runs', f'{i} {ds_name} {tq} 15min Flaml f', 15 * 60)

                values.append(y_and_prediction)

            
            
            



        # Maybe Change name when doing it finally for all datasets and tq -> will have 25 of those
        # adapt name properly of npz data



        np.savez(ds_name + ' dct ' + 'rsme_compRatio.npz', rmse_values=np.array(rmse_values), comp_ratios=np.array(comp_ratios))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run specific function.')
    parser.add_argument('function', type=str, choices=['tsf_or_f', 'best_time', 'test_deterministic', 'run_dataset'],
                        help='The function to run: tsf_or_f, best_time, test_deterministic, or run_dataset')
    
    args = parser.parse_args()
    
    if args.function == 'tsf_or_f':
        tsf_or_f()
    elif args.function == 'best_time':
        best_time()
    elif args.function == 'test_deterministic':
        test_deterministic()
    elif args.function == 'run_dataset':
        run_dataset()
    












    
