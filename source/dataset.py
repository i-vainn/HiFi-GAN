import librosa
import os
import random
import torch
import torchaudio

from torch import nn

from source.configs import ExperimentConfig
from source.configs import MelSpectrogramConfig


class VocDataset(torch.utils.data.Dataset):
    def __init__(self, train_config: ExperimentConfig, mel_config: MelSpectrogramConfig):
        self.config = train_config
        self.mel_config = mel_config
        self.index = os.listdir(os.path.join(train_config.data_dir))

        self.wav2spec = MelSpectrogram(mel_config)
        self.wav2spec.to(train_config.device)

    def pad_melspec(self, wav):
        padding = (self.mel_config.n_fft - self.mel_config.hop_length) // 2
        padded_wav = torch.nn.functional.pad(wav, (padding, padding), mode='reflect')

        return self.wav2spec(padded_wav)

    def segment_wav(self, wav):
        if wav.size(1) > self.config.segment_length:
            max_start = wav.size(1) - self.config.segment_length
            start = random.randint(0, max_start)
            wav = wav[:, start:start + self.config.segment_length]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.config.segment_length - wav.size(1)))
        
        return wav
    
    def __getitem__(self, idx):
        wav, _ = torchaudio.load(os.path.join(self.config.data_dir, self.index[idx]))
        wav = wav.to(self.config.device)
        wav = self.segment_wav(wav)
        mel = self.pad_melspec(wav)

        return wav, mel

    def __len__(self):
        return len(self.index)


class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            center=config.center
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel