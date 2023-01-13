
# Vehicle type recognition system - Pytorch implementation

The aim of the project was to create a neural
network model, which basing on the input data in the form
of an audio files would be able to recognize the type of the
vehicle passing.

## Installation

Install project with virtualenv

```bash
python [use versions 3.5-3.7] -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python .\InZnakExplorer.pyw
```
    
## Usage/Examples


```bash
python labelingUtilities.py combine logs/newLabels.csv logs/combinedLog.csv
python cutAudio.py combinedLog.csv
python main.py
```


## Documentation

### File structure

- InZnakExplorer.pyw - labeling utility supplied by Szymon Zaporowski
- cutAudio.py - a script that will split one long wav file with the recording into many individual wav files accoring to given log.csv file
- labelingUtilities - utilities for working with files with labels (eg. one-hot to num)
- results.txt - output file with test set evaluation metrics for latest traing
- labels.csv - final file with labels which is output from cutAudio.py script and input for the traing
- output.png - confussion matrix for latest traing
- requirements.txt - all Python libraries needed for script to work
- pythonVersionReq.txt - information about required Python version

#### logs
- combinedLog.csv - all logs combined in one file
- log*.csv - files with logs from labeling process that contains path to audio and video files, start and end of the track position in with vehicle class can be observed and class id in one-hot-encoding

#### trainingModel
- AudioClassifier.py - Pytorch model for audio classification
- AudioUtils.py - class with utilities for working with audio data (padding, sampling etc.)
- EarlyStopping.py - early stopping algorithm implementation
- main.py - main training and testing loop
- utilities.py - more general utilities

