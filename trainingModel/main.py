from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
from AudioClassifier import AudioClassifier
from AudioUtils import AudioUtils
from SoundDS import SoundDS
import utilities
import os
from EarlyStopping import EarlyStopping

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
def training(model, train_dl, validation_dl, num_epochs, early_stopping=None, results_csv_path='results.csv', training_id='tr_default', model_path='no_path'):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Loss Function, Optimizer and Scheduler
    learning_rate = 0.001
    criterion = torch.nn.CrossEntropyLoss()
    params = model.parameters()
    # optimizers = [
    #     torch.optim.Adadelta(params, lr=learning_rate),
    #     torch.optim.RMSprop(params, lr=learning_rate),
    #     torch.optim.SGD(params, lr=learning_rate),
    #     torch.optim.Adagrad(params, lr=learning_rate),
    #     torch.optim.Adam(params,lr=learning_rate)
    # ]
    optimizer = torch.optim.RMSprop(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    stop = False
    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # d, target = data
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
        acc_train = correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc_train:.2f}')

        # test early stopping and calc validation loss
        (valid_loss, acc_val), stop = inference(model, validation_dl, early_stopping)
        train_losses.append(avg_loss)
        val_losses.append(valid_loss)
        train_accs.append(acc_train)
        val_accs.append(acc_val)

        if stop:
            break

    print('Finished Training')

    # saving results
    correct_epoch_num = epoch
    if stop:
        correct_epoch_num -= 25

    best_train_loss = train_losses[correct_epoch_num]
    best_val_loss = val_losses[correct_epoch_num]
    best_train_acc = train_accs[correct_epoch_num]
    best_val_acc = val_accs[correct_epoch_num]
    save_training_results(results_csv_path, training_id, correct_epoch_num, best_val_loss, best_train_loss,
                          best_val_acc, best_train_acc, model_path)
    plot_training_history(train_losses, val_losses, train_accs, val_accs, correct_epoch_num, training_id, 'charts/')


def generate_training_id(epochs, train_loss, val_loss):
    str_tr = str(int(train_loss*1000))
    str_val = str(int(val_loss*1000))
    return str(epochs) + '_' + str_tr + '_' + str_val


def generate_chart_name(epochs, train_loss, val_loss, training_id=None):
    if training_id is None:
        training_id = generate_training_id(epochs, train_loss, val_loss)
    return 'train_chart_' + training_id + '.png'


def save_training_results(csv_path, training_id, epoch_num, val_loss, train_loss, val_acc, train_acc, model_path):
    is_file_exists = os.path.exists(csv_path)
    column_names = ['ID', 'epochs', 'val_loss', 'train_loss', 'val_acc', 'train_acc', 'model_path']
    if not is_file_exists:
        df = pd.DataFrame(columns=column_names)
    else:
        df = pd.read_csv(csv_path)

    row_df = pd.DataFrame([[training_id, epoch_num, val_loss, train_loss, val_acc, train_acc, model_path]],
                          columns=column_names)
    df = pd.concat([df, row_df])
    df.to_csv(csv_path, index=False)


def plot_training_history(train_losses, val_losses, train_accs, val_accs, stop_epoch_num, training_id=None, chart_dir=''):
    fig, (ax2, ax1) = plt.subplots(2, sharex=True)
    fig.suptitle("Training " + training_id)
    ax1.set_title('Loss')
    ax1.plot(val_losses, 'r', label='validation loss')
    ax1.plot(train_losses, 'b', label='training loss')
    ax1.plot([stop_epoch_num], [val_losses[stop_epoch_num]],  'ro')
    ax1.plot([stop_epoch_num], [train_losses[stop_epoch_num]], 'bo')
    ax1.axvline(x=stop_epoch_num, color='g', linestyle='--', label='stop')
    ax1.set(xlabel='Epoch', ylabel='Loss')
    ax1.legend()

    ax2.set_title('Accuracy')
    ax2.plot(val_accs, 'r', label='validation accuracy')
    ax2.plot(train_accs, 'b', label='training accuracy')
    ax2.plot([stop_epoch_num], [val_accs[stop_epoch_num]],  'ro')
    ax2.plot([stop_epoch_num], [train_accs[stop_epoch_num]], 'bo')
    ax2.axvline(x=stop_epoch_num, color='g', linestyle='--', label='stop')
    ax2.set(ylabel='Accuracy')
    ax2.legend()
    # plt.show()

    filename = generate_chart_name(stop_epoch_num, train_losses[stop_epoch_num], val_losses[stop_epoch_num], training_id)
    fig.savefig(chart_dir + filename)


# ----------------------------
# Inference
# ----------------------------
def inference (model, val_dl, early_stopping=None):
    correct_prediction = 0
    total_prediction = 0
    criterion = torch.nn.CrossEntropyLoss()
    valid_losses = []

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

            loss = criterion(outputs, labels)
            valid_losses.append(loss.item())

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
        
    acc = correct_prediction/total_prediction
    print(f'Validation set accuracy: {acc:.2f}, Total items: {total_prediction}')
    valid_loss = np.average(valid_losses)
    print(f'Validation loss: {valid_loss:.2f}')

    stats = valid_loss, acc
    if early_stopping is None:
        return stats, False

    early_stopping(valid_loss, model)
    if early_stopping.early_stop:
                print("STOP!")
                return stats, True
    return stats, False

if __name__ == '__main__':
    # paths
    dataset_location_path = 'D:/ProjektBadawczy/annotation/'
    audio_files_path = 'cutted_files/'
    full_audio_files_path = dataset_location_path + audio_files_path
    labels_csv_path = dataset_location_path + 'labels.csv'

    for i in range(5):
        training_id = '10n_4cnn_rmsprop_relu_' + str(i)
        model_dir = 'models/'
        model_filename = 'model_{0}.torch'.format(training_id)
        model_location_path = model_dir + model_filename

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

        early_stopping = EarlyStopping(patience=25, verbose=True)
        num_epochs=500
        training(myModel, train_data, val_data, num_epochs, early_stopping, training_id=training_id, model_path=model_location_path)
        myModel.load_state_dict(torch.load(early_stopping.path))
        myModel.eval()
        inference(myModel, val_data)
        inference(myModel, val_data)
        inference(myModel, val_data)
        inference(myModel, val_data)

        torch.save(myModel.state_dict(), model_location_path)
        #torch.save(myModel, model_location_path)


    # Old code

    # fig, (ax2, ax1) = plt.subplots(2, sharex=True)
    # #fig.title('Loss')
    # ax1.set_title('Loss')
    # ax1.plot([1, 2, 3], 'r', label='validation loss')
    # ax1.plot([4, 5, 6], 'g', label='training loss')
    # ax1.axvline(x=1, color='b', linestyle='--', label='stop')
    # ax1.set(ylabel='Loss')
    # ax1.legend()
    # ax2.set_title('Accuracy')
    # ax2.plot([10, 20, 30], 'r', label='validation loss')
    # ax2.plot([40, 50, 60], 'g', label='training loss')
    # ax2.set(xlabel='Epoch', ylabel='Accuracy')
    # ax2.legend()
    # plt.show()
    # fig.savefig('asd.png')

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