from os import listdir
from os.path import isfile, join
import pandas as pd
import pylab
import wave


def list_all_files(dir_path):
    return [f for f in listdir(dir_path) if isfile(join(dir_path, f))]


def write_filenames_to_csv(files_dir_path, csv_path):
    filenames = list_all_files(files_dir_path)
    df = pd.read_csv(csv_path)

    for i, filename in enumerate(filenames):
        df.at[i, 'filename'] = filename

    df.to_csv(csv_path, index=False)

def classnames_to_nums(csv_path):
    df = pd.read_csv(csv_path)
    for i, row in df.iterrows():
        if row['class'] in [0, 1, 2]:
            continue
        elif row['class'] == 'motorcycle':
           df.at[i,'class'] = 2
        elif row['class'] == 'truck':
           df.at[i,'class'] = 1
        else:
           df.at[i,'class'] = 0
    df.to_csv(csv_path, index=False)

    
def graph_spectrogram(wav_file):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig('D:/ProjektBadawczy/annotation/spectrogram.png')


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate
