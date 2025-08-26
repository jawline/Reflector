import torch
from ddpm_scheduler import DDPM_Scheduler
from unet import UNET
from torch.optim import Adam
from timm.utils import ModelEmaV3
from labml_nn.diffusion.stable_diffusion.model.autoencoder import (
    Autoencoder,
    Encoder,
    Decoder,
)


def create_autoencoder(device, lr):
    z_channels = 2048
    emb_channels = 8
    encoder = Encoder(
        in_channels=1,
        z_channels=z_channels,
        channels=32,
        channel_multipliers=[4, 2, 8, 8, 2],
        n_resnet_blocks=1,
    ).to(device)
    decoder = Decoder(
        out_channels=1,
        z_channels=z_channels,
        channels=32,
        channel_multipliers=[4, 2, 8, 8, 2],
        n_resnet_blocks=1,
    ).to(device)
    optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    autoencoder = Autoencoder(
        encoder, decoder, emb_channels=emb_channels, z_channels=z_channels
    ).to(device)
    return autoencoder, optimizer


def load_autoencoder(device, file, lr):
    autoencoder, optimizer = create_autoencoder(device, lr)
    try:
        print("Trying to load", file)
        if file is not None:
            checkpoint = torch.load(file, map_location=device)
            print("Loaded checkpoint")
            autoencoder.load_state_dict(checkpoint["autoencoder"])
            print("Loaded model")
            optimizer.load_state_dict(checkpoint["optimizer"])
    except Exception as e:
        print("Could not load checkpoint", e)

    return autoencoder, optimizer


def create(device, ema_decay, num_time_steps, lr):
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps, device=device)
    model = UNET(time_steps=num_time_steps).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay).to(device)
    scaler = torch.amp.GradScaler("cuda")
    return scheduler, model, optimizer, ema, scaler


def load(device, file, ema_decay, num_time_steps, lr):
    scheduler, model, optimizer, ema, scaler = create(
        device, ema_decay, num_time_steps, lr
    )
    try:
        print("Trying to load", file)
        if file is not None:
            checkpoint = torch.load(file, map_location=device)
            print("Loaded checkpoint")
            model.load_state_dict(checkpoint["weights"])
            print("Loaded model")
            ema.load_state_dict(checkpoint["ema"])
            print("Loaded ema")
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded optimizer")
            scaler.load_state_dict(checkpoint["scaler"])
            print("Loaded scaler")
    except Exception as e:
        print("Could not load checkpoint", e)

    return scheduler, model, optimizer, ema, scaler
