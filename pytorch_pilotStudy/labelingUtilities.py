import pandas as pd
import sys
from datetime import datetime

def absolute_path_to_relative(csv_path):
    
    input_dataframe = pd.read_csv(csv_path)
    columns_names = ['audio_file_name', 'video_file_name']
    
    for i, row in input_dataframe.iterrows():
        for col_name in columns_names:
            abs_paths = row[col_name]
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
    new_file_names = ['video_file_name', 'audio_file_name']
    new_time_names = ['video_position', 'audio_position_start', 'audio_position_end']
    new_class_names = [ 'motorcycle_present', 'car_present', 'van_present', 'truck_present', 'bus_present']
    all_column_names = (new_file_names + new_time_names + new_class_names)
    
    new_dataframe = pd.DataFrame(columns=all_column_names)
    
    for i, row in df.iterrows():
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
        row = pd.DataFrame([['', audio_path, '', time_start, time_end, *vehicle_present]],
                           columns=all_column_names)
        new_dataframe = new_dataframe.append(row)
        
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
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit()
        
    if sys.argv[1] == 'torelative':
        absolute_path_to_relative('./log.csv')
    if sys.argv[1] == 'tonewlog':
        old_log_to_new_log('./Audio_DATA_20190524T1400V1200A.csv', 'audio_data_20190524T120000.wav')

    