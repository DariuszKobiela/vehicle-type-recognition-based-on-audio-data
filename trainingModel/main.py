import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
from AudioClassifier import AudioClassifier
from AudioUtils import AudioUtils
from SoundDS import SoundDS
import utilities
import os

from scipy import signal
from scipy.io import wavfile
import numpy as np


def prepare_data(df, data_path, train_size=0.8):
    myds = SoundDS(df, data_path)

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * train_size)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
    return train_dl, val_dl


# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, validation_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            #if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
        
        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
        inference(model, validation_dl)

    print('Finished Training')


# ----------------------------
# Inference
# ----------------------------
def inference (model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
        
    acc = correct_prediction/total_prediction
    print(f'Validation set accuracy: {acc:.2f}, Total items: {total_prediction}')


if __name__ == '__main__':

    # paths
    dataset_location_path = 'D:/ProjektBadawczy/annotation/'
    model_location_path = dataset_location_path + 'model.torch'
    audio_files_path = 'cutted_files/'
    full_audio_files_path = dataset_location_path + audio_files_path
    labels_csv_path = dataset_location_path + 'labels.csv'

    df = pd.read_csv(labels_csv_path)
    train_data, val_data = prepare_data(df, dataset_location_path)
    
    myModel = AudioClassifier()
    if os.path.exists(model_location_path):
        myModel.load_state_dict(torch.load(model_location_path))
        myModel.eval()

    # Put model on the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    myModel = myModel.to(device)
    # Check that it is on Cuda
    next(myModel.parameters()).device

    num_epochs=400   # Just for demo, adjust this higher.
    training(myModel, train_data, val_data, num_epochs)
    inference(myModel, val_data)

    torch.save(myModel.state_dict(), dataset_location_path + 'model.torch')
    #torch.save(myModel, dataset_location_path + 'model')


    # Old code

    # audio_file  = AudioUtils.open(full_audio_files_path + 'VehicleNoise0.wav')
    # audio_file = AudioUtils.rechannel(audio_file, 1)

    # samples, sample_rate = audio_file
    # frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    # plt.pcolormesh(times, frequencies, spectrogram)
    # plt.imshow(spectrogram)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

    # utilities.graph_spectrogram(full_audio_files_path + 'VehicleNoise0.wav')

    # utilities.write_filenames_to_csv(full_audio_files_path, labels_csv_path)
    # utilities.classnames_to_nums(labels_csv_path)

    # df = pd.read_csv(labels_csv_path)
    # sound_ds = SoundDS(df, dataset_location_path)
    # spec, classid = sound_ds[0]
    # print(classid)
    # print(spec.shape)
    
    # plt.figure()
    # plt.imshow(spec[0].t().numpy(), aspect='auto', origin='lower')
    # plt.show()

    # AudioUtils.plot_spectrogram(spec[0])

    # audio_file  = AudioUtils.open(full_audio_files_path + 'VehicleNoise0.wav')
    # audio_file = AudioUtils.rechannel(audio_file, 2)
    # audio_file = AudioUtils.pad_trunc(audio_file, 6000)

    # AudioUtils.save(audio_file, full_audio_files_path + 'newAudio.wav')
    # data_waveform, rate_of_sample = audio_file
    # print(data_waveform.shape[0])


    # plt.figure()
    # plt.plot(data_waveform.t().numpy())
    # plt.show()