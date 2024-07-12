import argparse
from flaml_and_fwiz import run_flaml
from utils.personal_utils import *

path = '/home/simon/TSER/preparedData/'

path_ts = '/home/simon/TSER/Time_Series_Data/'


# For Agluon change all occurences
# For new Dataset, change ds_name
# Maybe add parameter option to do all test at once! -> delete before that the old experiments!
ds_name = 'AppliancesEnergy'



def tsf_or_f():

    ds_names = ['FloodModeling1', 'Covid3Month', 'AppliancesEnergy']

    for ds_name in ds_names:

        for time in [15, 30]:
            source_path_tsf = path + ds_name + '_TRAIN_None__ts_and_features.csv'
            source_path_f = path + ds_name + '_TRAIN_None__features.csv'

            run_flaml(source_path_tsf, 'Test if tsf or f', f'tsf {ds_name} Flaml {time}min', time * 60)
            run_flaml(source_path_f, 'Test if tsf or f', f'f {ds_name} Flaml {time}min', time * 60)



def best_time():
    # Letting run only with features
    
    ds_names = ['FloodModeling1', 'Covid3Month', 'AppliancesEnergy']
    times = [1 ,5, 15, 20, 25, 30, 60]

    for ds_name in ds_names:

        source_path = path + ds_name + '_TRAIN_None__ts_and_features.csv'


        for time in times:
            run_flaml(source_path, 'Test best runtime', f'{time}min {ds_name} Flaml tsf', time * 60)



def test_deterministic():
    source_path = path + ds_name + '_TRAIN_None__features.csv'

    for i in range(3):
        run_flaml(source_path, 'Test deterministic', f'Run{i+1} Appliances Flaml 5min', 5 * 60)




# I assume only f, 15min -> maybe change later


# Do the runs, put on mlflow
# Then calculte comp_ratios and save it as array -> transfer back, from that I can do bar chart!
def testrun_dct():

    rmse_values = []
    comp_ratios = []

    for i in [0.5,0.75,0.85,0.95,0.99]:
        source_path = path + ds_name + '_TRAIN' + '_dct' + f'_{i}' + '_features.csv'

        rmse = run_flaml(source_path, 'Appliances dct test run', f'{i} dct Appliances Flaml', 15 * 60)


        rmse_values += rmse

        # Get the comp ratio of TEST-SET and of TRAIN SET, then add and divide by 2 -> Compression Ratio of both sets! More accurate than only one!
        ts_name = 'AppliancesEnergy_TRAIN.ts'
        data_path_ts = path_ts + ts_name


        comp_Ratio_train = get_compratio(data_path_ts, 'dct', i)
        comp_Ratio_test = get_compratio(data_path_ts.replace('TRAIN','TEST'), 'dct', i)

        comp_ratios += (comp_Ratio_train + comp_Ratio_test) / 2

    
    # Maybe Change name when doing it finally for all datasets and tq -> will have 25 of those
    np.savez('rsme_compRatio.npz', array1=np.array(rmse_values), array2=np.array(comp_ratios))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run specific function.')
    parser.add_argument('function', type=str, choices=['tsf_or_f', 'best_time', 'test_deterministic'],
                        help='The function to run: tsf_or_f, best_time, or test_deterministic')
    
    args = parser.parse_args()
    
    if args.function == 'tsf_or_f':
        tsf_or_f()
    elif args.function == 'best_time':
        best_time()
    elif args.function == 'test_deterministic':
        test_deterministic()

    












    
