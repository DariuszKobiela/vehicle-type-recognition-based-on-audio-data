The aim of the project was to classify the type of vehicle, based on its sound

There are three classes of vehicles
- Cars
- Trucks, buses and vans
- Motorcycles

This implementation uses two types of representation of audio data
- Spectrograms
- MFCCs

All samples should have the same duration.
In the preprocessing stage, we cut files that are longer than 6 seconds and we added silence to files that are shorter than 6 seconds.

The model input data are created form the cutted files. They come in the form of Spectrograms and MFCCs.
Different types and architectures of convolutional neural networks have been tested, but the current one performed best.
Training for each type of input was performed approximately 10 times. Each time the results were similar.

We achived the accurace of 0.87 using spectrograms.

The results of using spectrograms and MFCCs are very similar. But the spectrogram results are slightly better.
