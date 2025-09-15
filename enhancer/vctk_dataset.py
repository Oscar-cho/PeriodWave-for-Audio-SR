import os
import sys
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from glob import glob

class VCTKDataset(Dataset):
    def __init__(self, vctk_path, data_config, segment_size=8192, train=True):
        self.vctk_path = vctk_path
        self.segment_size = segment_size
        self.sampling_rate_hr = data_config['sampling_rate'] # e.g., 24000
        # Use floor division for sr_factor
        self.sampling_rate_lr = data_config['sampling_rate'] // data_config.get('sr_factor', 2)
        self.train = train

        self.h = data_config

        # Use glob to find all .wav files recursively
        self.wav_files = glob(os.path.join(self.vctk_path, '**', '*.flac'), recursive=True)
        
        # Simple train/val split
        if not self.wav_files:
            raise ValueError(f"No .flac files found in the directory: {self.vctk_path}")
            
        if train:
            self.wav_files = self.wav_files[:-100]
        else:
            self.wav_files = self.wav_files[-100:]

        self.resampler_lr = torchaudio.transforms.Resample(
            orig_freq=self.sampling_rate_hr,
            new_freq=self.sampling_rate_lr
        )

        self.resampler_hr = torchaudio.transforms.Resample(
            orig_freq=self.sampling_rate_lr,
            new_freq=self.sampling_rate_hr
        )

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]

        # Load HR audio
        try:
            audio_hr, sr = torchaudio.load(wav_path)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.wav_files))

        if sr != self.sampling_rate_hr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate_hr)
            audio_hr = resampler(audio_hr)

        # Mono channel
        if audio_hr.size(0) > 1:
            audio_hr = torch.mean(audio_hr, dim=0, keepdim=True)

        # Normalize to [-1, 1] range
        if torch.abs(audio_hr).max() > 0:
            audio_hr = audio_hr / torch.abs(audio_hr).max() * 0.95
        else:
            # Handle silence by skipping
            return self.__getitem__((idx + 1) % len(self.wav_files))

        # Create LR audio by downsampling
        audio_lr = self.resampler_lr(audio_hr)

        # Pad if the audio is too short
        sr_factor = self.sampling_rate_hr // self.sampling_rate_lr
        if audio_hr.size(1) < self.segment_size:
            audio_hr = torch.nn.functional.pad(audio_hr, (0, self.segment_size - audio_hr.size(1)))

        min_lr_len = self.segment_size // sr_factor
        if audio_lr.size(1) < min_lr_len:
            audio_lr = torch.nn.functional.pad(audio_lr, (0, min_lr_len - audio_lr.size(1)))


        # Randomly crop to segment size
        if self.train and audio_hr.size(1) > self.segment_size:
            start_frame_hr = random.randint(0, audio_hr.size(1) - self.segment_size)
            start_frame_lr = start_frame_hr // sr_factor

            audio_hr_seg = audio_hr[:, start_frame_hr : start_frame_hr + self.segment_size]
            audio_lr_seg = audio_lr[:, start_frame_lr : start_frame_lr + (self.segment_size // sr_factor)]
        else:
            audio_hr_seg = audio_hr[:, :self.segment_size]
            audio_lr_seg = audio_lr[:, :(self.segment_size // sr_factor)]


        # Generate mel spectrograms
        mel_hr = self.get_mel(audio_hr_seg)

        audio_lr_upsampled = self.resampler_hr(audio_lr_seg)

        if audio_lr_upsampled.size(1) > audio_hr_seg.size(1):
            audio_lr_upsampled = audio_lr_upsampled[:, :audio_hr_seg.size(1)]
        else:
            audio_lr_upsampled = torch.nn.functional.pad(audio_lr_upsampled, (0, audio_hr_seg.size(1) - audio_lr_upsampled.size(1)))

        mel_lr = self.get_mel(audio_lr_upsampled)

        return mel_lr.squeeze(0), mel_hr.squeeze(0)

    def get_mel(self, audio):
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.h['sampling_rate'],
            n_fft=self.h['n_fft'],
            win_length=self.h['win_length'],
            hop_length=self.h['hop_length'],
            n_mels=self.h['n_mel_channels'],
            f_min=self.h['mel_fmin'],
            f_max=self.h['mel_fmax'],
            power=1.0,
            norm='slaney',
            mel_scale='slaney'
        )

        mel = mel_spectrogram_transform(audio)

        # Log-scale
        mel = torch.log(torch.clamp(mel, min=1e-5))

        return mel
