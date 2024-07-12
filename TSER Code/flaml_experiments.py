import argparse
from flaml_and_fwiz import run_flaml

path = '/home/simon/TSER/preparedData/'


# Add List and Loop for doing more datasets at once
ds_name = 'AppliancesEnergy'



def tsf_or_f():
    source_path_tsf = path + ds_name + '_TRAIN_None__ts_and_features.csv'
    source_path_f = path + ds_name + '_TRAIN_None__features.csv'

    run_flaml(source_path_tsf, 'Test if tsf or f', 'Appliances Flaml tsf 15min', 15 * 60)
    run_flaml(source_path_f, 'Test if tsf or f', 'Appliances Flaml f 15min', 15 * 60)



def best_time():
    # Letting run only with features
    source_path = path + ds_name + '_TRAIN_None__features.csv'

    times = [1 ,5, 15, 30, 60]
    for time in times:
        run_flaml(source_path, 'Test best runtime', f'Appliances Flaml {time}min', time * 60)


# run 3 times the same test
def test_deterministic():
    source_path = path + ds_name + '_TRAIN_None__features.csv'

    for i in range(3):
        run_flaml(source_path, 'Test deterministic', f'Run{i+1} Appliances Flaml 5min', 5 * 60)





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

    












    
