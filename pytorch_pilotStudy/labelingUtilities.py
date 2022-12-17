import pandas as pd
import sys
from datetime import datetime

csv_col_file_names = ['video_file_name', 'audio_file_name']
csv_col_time_names = ['video_position', 'audio_position_start', 'audio_position_end']
csv_col_class_names = [ 'motorcycle_present', 'car_present', 'van_present', 'truck_present', 'bus_present']

def absolute_path_to_relative(csv_path):
    
    input_dataframe = pd.read_csv(csv_path)
    columns_names = ['audio_file_name', 'video_file_name']
    
    for i, row in input_dataframe.iterrows():
        for col_name in columns_names:
            abs_paths = row[col_name]
            if not isinstance(abs_paths, str):
                continue
            abs_paths_split = abs_paths.split('/')
            relative_path = 'data/' + abs_paths_split[-1]
            input_dataframe.at[i, col_name] = relative_path
            
    input_dataframe.to_csv(csv_path, index=False)
    

def old_log_to_new_log(csv_path, audio_path):
    # set time frame window length [ms]
    time_frame = 4000
    time_move = time_frame / 2
    
    df = pd.read_csv(csv_path)
    
    # old file format columns
    col_time_name = 'timestamp'
    col_class_names = ['motorcycle', 'car', 'van', 'truck', 'bus']
    
    # new file format columns
    all_column_names = (csv_col_file_names + csv_col_time_names + csv_col_class_names)
    
    new_dataframe = pd.DataFrame(columns=all_column_names)
    
    for _, row in df.iterrows():
        # get time
        timestamp = row[col_time_name]
        ms_time = time_to_ms(timestamp)
        if ms_time < time_move:
            time_start = 0
        else:
            time_start = ms_time - time_move
        time_end = ms_time + time_move
        
        # get class
        vehicle_present = [0, 0, 0, 0, 0]
        for class_idx, class_name in enumerate(col_class_names):
            if row[class_name] == 1:
                vehicle_present[class_idx] = 1
                break   # take just one (the first one) for single class clasification
        
        # no class found then skip
        if not any(vehicle_present):
            continue
        
        # save to df
        new_row = pd.DataFrame([['', audio_path, '', int(time_start), int(time_end), *vehicle_present]],
                           columns=all_column_names)
        new_dataframe = new_dataframe.append(new_row)
        
    # save csv
    save_csv_path = 'test_oldLog.csv'
    new_dataframe.to_csv(save_csv_path, index=False)
        

def time_to_ms(str_time):
    time_split = str_time.split(':')
    ms = int(time_split[0]) * 60 * 60 * 1000 # h
    ms = ms + int(time_split[1]) * 60 * 1000 # min
    ms = ms + int(time_split[2]) * 1000 # sec
    ms = ms + int(time_split[3]) # ms
    return ms


def combine_csvs(*csv_paths, save_csv_path='combinedLog.csv'):
    dfs = [pd.read_csv(csv_path) for csv_path in csv_paths]
    new_df = pd.concat(dfs)
    
    # save csv
    new_df.to_csv(save_csv_path, index=False)
    
    
if __name__ == '__main__':
    arg_num = len(sys.argv)
    if arg_num < 2:
        print('Not enough arguments')
        sys.exit()
    action = sys.argv[1]
        
    if action == 'torelative':
        if arg_num != 3:
            print('Wrong num of arguments - csv_path')
            sys.exit()
        absolute_path_to_relative(sys.argv[2])
        
    if action == 'tonewlog':
        if arg_num != 4:
            print('Wrong num of arguments - csv_path, wav_path')
            sys.exit()
        old_log_to_new_log(sys.argv[2], sys.argv[3])
        
    if action == 'combine':
        if arg_num < 4:
            print('Not enough arguments - *csv_paths')
            sys.exit()
        combine_csvs(*sys.argv[2:])
        

    