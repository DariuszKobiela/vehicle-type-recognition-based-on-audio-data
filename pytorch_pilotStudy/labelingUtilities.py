import pandas as pd
import sys

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
        
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit()
        
    if sys.argv[1] == 'torelative':
        absolute_path_to_relative('./log.csv')

    