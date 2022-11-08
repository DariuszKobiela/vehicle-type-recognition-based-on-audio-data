from pydub import AudioSegment
import pandas as pd
import os
import sys


def cut_audio_based_on_csv(csv_path, audio_save_path='', labels_save_path=''):

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
        label = 0
        if row['motorcycle_present'] == 1 or row['bus_present'] == 1 or row['van_present'] == 1:
            label = 2
        elif row['truck_present'] == 1:
            label = 1
        elif row['car_present'] == 1:
            label = 0
        else:
            label = -1

        # append new row
        row = pd.DataFrame([[out_audio_path, label]], columns=['file_path', 'class'])
        output_dataframe = output_dataframe.append(row)
        
        # remember last file to not open it again if next is the same
        last_audio_path = audio_path
    
    # save csv
    save_csv_path = labels_save_path + 'labels.csv'
    output_dataframe.to_csv(save_csv_path, index=False)


def create_csv_with_labels(in_csv_path, out_csv_path):
    df = pd.DataFrame()


if __name__ == '__main__':
    csv_path = 'log.csv' # default
    arg_num = len(sys.argv)
    if arg_num == 2:
        csv_path = sys.argv[1]
    
    save_path = 'cutted_files/'
    cut_audio_based_on_csv(csv_path, save_path)