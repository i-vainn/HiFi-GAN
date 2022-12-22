import os
import wget
import torch
import torchaudio

from source.generator_model import Generator
from source.dataset import MelSpectrogram
from source.configs import MelSpectrogramConfig

CHECKPOINT_URL = 'https://api.wandb.ai/files/i_vainn/hifigan/anzmldlc/checkpoints-V1/checkpoint_16ep.pth?_gl=1*16cg6xj*_ga*MTc4NzY3MDAxNy4xNjcwMjQwMjc2*_ga_JH1SJHJQXJ*MTY3MTczNTA3MC41MS4xLjE2NzE3MzUwODUuNDUuMC4w'
CHECKPOINT_PATH = 'model_best.pth'
TEST_PATH = 'data/test/wavs'
OUTPUT_PATH = 'data/test/generated_wavs'
DEVICE = 'cpu'

if __name__ == '__main__':
    wget.download(CHECKPOINT_URL, CHECKPOINT_PATH)
    generator = Generator()
    generator.to(DEVICE)
    generator.load_state_dict(torch.load(CHECKPOINT_PATH)['generator'])
    melspec = MelSpectrogram(MelSpectrogramConfig())
    generator.eval()

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print('\nGenerating...')

    with torch.inference_mode():
        for file in os.listdir(TEST_PATH):
            original_wav, sr = torchaudio.load(os.path.join(TEST_PATH, file))
            original_mel = melspec(original_wav).to(DEVICE)
            generated_wav = generator(original_mel)
            torchaudio.save(os.path.join(OUTPUT_PATH, file), generated_wav, sr)
    
    print('Done!')