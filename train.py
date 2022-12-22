import os
import torch
import wandb

from tqdm import tqdm

from source.utils import seed_everything
from source.utils import collate_fn
from source.configs import ExperimentConfig
from source.configs import MelSpectrogramConfig
from source.dataset import VocDataset
from source.training import evaluate
from source.training import train_epoch
from source.generator_model import Generator
from source.discriminator_model import SuperDiscriminator


def main(config: ExperimentConfig):
    train_dataset = VocDataset(config, MelSpectrogramConfig())
    generator = Generator().to(config.device)
    discriminator = SuperDiscriminator().to(config.device)

    generator_opt = torch.optim.AdamW(generator.parameters(), lr=config.lr, betas=config.betas)
    discriminator_opt = torch.optim.AdamW(discriminator.parameters(), lr=config.lr, betas=config.betas)

    generator_sched = torch.optim.lr_scheduler.ExponentialLR(generator_opt, config.sched_decay)
    ddiscriminator_sched = torch.optim.lr_scheduler.ExponentialLR(discriminator_opt, config.sched_decay)

    dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    wandb.init(
        project=config.project_name,
        entity=config.entity,
        config=config.to_dict(),
    )
    os.makedirs(config.save_dir, exist_ok=True)

    for epoch in tqdm(range(1, config.n_epochs + 1), desc='Epochs', total=config.n_epochs+1):
        train_epoch(
            generator, discriminator,
            dataloader, generator_opt, discriminator_opt,
            generator_sched, ddiscriminator_sched,
            config
        )
        evaluate(generator, config.test_path, config.device)

        if epoch % config.save_epochs == 0:
            print('Saving new checkpoint on epoch {}...'.format(epoch))
            save_path = os.path.join(config.save_dir, f"checkpoint_{epoch}ep.pth")
            torch.save({
                'generator': generator.state_dict(), 
                'discriminator': discriminator.state_dict()
                }, save_path
            )
            wandb.save(save_path)

    wandb.finish()


if __name__ == "__main__":
    config = ExperimentConfig()
    seed_everything(config.seed)

    main(config)