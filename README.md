# HiFi-GAN
## Translate mel-spectrograms into audio using Generative Adversarial Network

This is unofficial simple HiFi-GAN implementation. 
You can either use pre-trained weights (however this implies some artifacts in the generated audio)
or train your own GAN.
You can find training logs, weights and
some synthesized examples [here](https://wandb.ai/i_vainn/hifigan/reports/HiFi-GAN-Implementation-Report--VmlldzozMjAxMjcx)

Installation:

```bash
git clone https://github.com/ivan7022/HiFi-GAN.git

cd HiFi-GAN

pip install -r requirements.txt
```

To download and apply pre-trained model you simply run

```bash
python inference.py
```

Change some path variables or device (by default model runs on CPU) if you need.

To reproduce results / train your own GAN you should download data (by default it is LJSpeech-1.1) and run training:

```bash
bash setup_data.sh

python train.py
```

Do not forget to change wandb logging variables in `source/config.py` in this case.
