from torch.utils.data import Dataset
from AudioUtils import AudioUtils

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
  def __init__(self, df, data_path):
    self.df = df
    self.data_path = str(data_path)
    self.duration = 5000
    self.sr = 48000
    self.channel = 2
    self.shift_pct = 0.4
            
  # ----------------------------
  # Number of items in dataset
  # ----------------------------
  def __len__(self):
    return len(self.df)    
    
  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
  def __getitem__(self, idx):
    # Absolute file path of the audio file - concatenate the audio directory with
    # the relative path
    audio_file = self.data_path + self.df.loc[idx, 'file_path']
    # Get the Class ID
    class_id = self.df.loc[idx, 'class']

    aud = AudioUtils.open(audio_file)
    # Some sounds have a higher sample rate, or fewer channels compared to the
    # majority. So make all sounds have the same number of channels and same 
    # sample rate. Unless the sample rate is the same, the pad_trunc will still
    # result in arrays of different lengths, even though the sound duration is
    # the same.
    reaud = AudioUtils.resample(aud, self.sr)
    rechan = AudioUtils.rechannel(reaud, self.channel)

    dur_aud = AudioUtils.pad_trunc(rechan, self.duration)
    shift_aud = AudioUtils.time_shift(dur_aud, self.shift_pct)
    sgram = AudioUtils.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = AudioUtils.spectro_augment(sgram, max_mask_pct=0.06, n_freq_masks=2, n_time_masks=2)

    return aug_sgram, class_id
  
  
  def get_item_with_no_aug(self, idx):
    # Absolute file path of the audio file - concatenate the audio directory with
    # the relative path
    audio_file = self.data_path + self.df.loc[idx, 'file_path']
    # Get the Class ID
    class_id = self.df.loc[idx, 'class']

    aud = AudioUtils.open(audio_file)
    # Some sounds have a higher sample rate, or fewer channels compared to the
    # majority. So make all sounds have the same number of channels and same 
    # sample rate. Unless the sample rate is the same, the pad_trunc will still
    # result in arrays of different lengths, even though the sound duration is
    # the same.
    reaud = AudioUtils.resample(aud, self.sr)
    rechan = AudioUtils.rechannel(reaud, self.channel)

    dur_aud = AudioUtils.pad_trunc(rechan, self.duration)
    shift_aud = dur_aud
    sgram = AudioUtils.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = sgram

    return aug_sgram, class_id