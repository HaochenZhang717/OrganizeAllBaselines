import torch
from Utils.io_utils import instantiate_from_config


def build_dataloader(config, args=None):
    batch_size = config['dataloader']['batch_size']
    # Pass save_dir to dataset if needed (harmless if ignored)
    if args is not None:
        config['dataloader']['train_dataset']['params']['output_dir'] = args.save_dir
    train_dataset = instantiate_from_config(config['dataloader']['train_dataset'])
    valid_dataset = instantiate_from_config(config['dataloader']['valid_dataset'])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    return {
        'train_dataloader': train_dataloader,
        'train_dataset':    train_dataset,
        'valid_dataloader': valid_dataloader,
        'valid_dataset':    valid_dataset,
    }
