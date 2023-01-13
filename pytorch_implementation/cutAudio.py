from pydub import AudioSegment
import pandas as pd
import os
import sys
from os import listdir
from os.path import isfile, join
from labelingUtilities import combine_csvs


def cut_audio_based_on_csv(csv_path, class_row_names, audio_save_path='', labels_save_path=''):

    # load files
    last_audio_path = None
    input_dataframe = pd.read_csv(csv_path)

    output_dataframe = pd.DataFrame(columns=['file_path', 'class'])

    # create output dirs if necessary
    for dir in [audio_save_path, labels_save_path]:
        if not dir:
            continue
        is_savedir_exists = os.path.exists(dir)
        if not is_savedir_exists:
            os.makedirs(dir)

    # cut and save new files
    for i, row in input_dataframe.iterrows():
        audio_path = row['audio_file_name']
        t1 = row['audio_position_start']
        t2 = row['audio_position_end']
        
        if audio_path != last_audio_path:
            audio_file = AudioSegment.from_wav(audio_path)
        
        cutted = audio_file[t1:t2]
        out_audio_path = audio_save_path + 'VehicleNoise{0}.wav'.format(i)
        cutted.export(out_audio_path, format='wav')

        # add new file to labels file
        label = -1
        for row_name in class_row_names:
            if row[row_name] == 1:
                label = classname_to_int(row_name, class_row_names)
                if label == -1:
                    print('Warning: No label given for row number: ' + str(i))
        
        
        # if row['motorcycle_present'] == 1:
        #     label = 2
        # elif row['truck_present'] == 1 or row['bus_present'] == 1 or row['van_present'] == 1:
        #     label = 1
        # elif row['car_present'] == 1:
        #     label = 0
        # else:
        #     label = -1
        #     print('Warning: No label given for row number: ' + str(i))

        # append new row
        row = pd.DataFrame([[out_audio_path, label]], columns=['file_path', 'class'])
        output_dataframe = output_dataframe.append(row)
        
        # remember last file to not open it again if next is the same
        last_audio_path = audio_path
    
    # save csv
    save_csv_path = labels_save_path + 'labels.csv'
    output_dataframe.to_csv(save_csv_path, index=False)


def classname_to_int(name, class_names):
    for i, class_name in enumerate(class_names):
        if name == class_name:
            return classid_to_combined_classid(i)
    print('Warning: Unknown class!')
    return -1


def classid_to_combined_classid(class_id):
    # ['car', 'truck', 'motorcycle', 'van', 'bus']
    if (class_id == 3 or class_id == 4): # van or bus
        class_id = 1 # add to trucks
    return class_id


def dir_to_classes(dir_path, class_names):
    output_dataframe = pd.DataFrame(columns=['file_path', 'class'])
    for f in listdir(dir_path):
        # only dirs
        subdir = join(dir_path, f)
        if isfile(subdir):
            continue 
        
        label = classname_to_int(f, class_names)
        if label == -1:
            continue
        
        for audio_file in listdir(subdir):
            row = pd.DataFrame([['data/additional/' + str(f) + '/' + str(audio_file), label]], columns=['file_path', 'class'])
            output_dataframe = output_dataframe.append(row)
            
    output_dataframe.to_csv('newLabels.csv', index=False)
                

    
if __name__ == '__main__':
    csv_path = 'log.csv' # default
    arg_num = len(sys.argv)
    if arg_num == 2:
        csv_path = sys.argv[1]
    
    new_files_dir_path = 'data\\additional'
    save_path = 'cutted_files/'
    # class_row_names = ['car_present', 'truck_present', 'motorcycle_present', 'van_present', 'bus_present']
    class_names = ['car', 'truck', 'motorcycle', 'van', 'bus']
    class_row_names = [name + '_present' for name in class_names]
    cut_audio_based_on_csv(csv_path, class_row_names, save_path)
    dir_to_classes(new_files_dir_path, class_names)
    combine_csvs('newLabels.csv', 'labels.csv', save_csv_path='labels.csv')
