import wandb
import os
import torch
from torch import nn
from tqdm import tqdm

from source.utils import AverageMeter


def train_epoch(
    generator,
    discriminator,
    train_dataloader,
    gopt,
    dopt,
    gen_scheduler,
    disc_scheduler,
    config,
):
    generator.train()

    dtotal_meter = AverageMeter()
    gfm_meter = AverageMeter()
    gl1_meter = AverageMeter()
    gtotal_meter = AverageMeter()
    gadv_meter = AverageMeter()
    log_steps = 20

    for step, (wav, mel) in enumerate(tqdm(train_dataloader), 1):
        wav = wav.to(config.device)
        mel = mel.to(config.device)

        generated_wav = generator(mel)
        generated_mel = train_dataloader.dataset.pad_melspec(generated_wav)

        dopt.zero_grad()
        disc_real, _ = discriminator(wav)
        disc_fake, _ = discriminator(generated_wav.detach())
        disc_loss = ((disc_real - 1) ** 2).mean() + (disc_fake ** 2).mean()
        disc_loss.backward()
        dopt.step()

        disc_real, disc_real_acts = discriminator(wav)
        disc_fake, disc_fake_acts = discriminator(generated_wav)

        matching_loss = nn.functional.l1_loss(disc_real_acts, disc_fake_acts) * config.matching_gamma
        l1_loss = nn.functional.l1_loss(mel, generated_mel) * config.l1_gamma
        adv_loss = ((disc_fake - 1) ** 2).mean() * config.adv_gamma
        generator_loss = matching_loss + l1_loss + adv_loss

        gopt.zero_grad()
        generator_loss.backward()
        gopt.step()

        gl1_meter.update(l1_loss.item(), wav.size(0))
        gfm_meter.update(matching_loss.item(), wav.size(0))
        gtotal_meter.update(generator_loss.item(), wav.size(0))
        dtotal_meter.update(disc_loss.item(), wav.size(0))
        gadv_meter.update(adv_loss, wav.size(0))

        if step % log_steps == 0:
            wandb.log({
                'train/total generator loss': gtotal_meter.avg,
                'train/l1 generator loss': gl1_meter.avg,
                'train/matching generator loss': gfm_meter.avg,
                'train/adversarial generator loss': gadv_meter.avg,
                'train/total discriminator loss': dtotal_meter.avg,
                'train/learning rate': gopt.param_groups[0]['lr']
            })

            gtotal_meter.reset()
            gl1_meter.reset()
            gfm_meter.reset()
            gadv_meter.reset()
            dtotal_meter.reset()

    disc_scheduler.step()
    gen_scheduler.step()


@torch.inference_mode()
def evaluate(generator, test_path, device):
    generator.eval()

    for file in os.listdir(test_path):
        mel = torch.load(os.path.join(test_path, file)).unsqueeze(0).to(device)
        wav = generator(mel).squeeze(0).detach().cpu()
        wandb.log({
            "test/" + file[:-3]: wandb.Audio(wav, 22050),
        })