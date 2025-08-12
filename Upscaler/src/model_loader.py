import torch
from ddpm_scheduler import DDPM_Scheduler
from unet import UNET
from torch.optim import Adam
from timm.utils import ModelEmaV3


def create(device, ema_decay, num_time_steps, lr):
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = UNET().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay).to(device)
    scaler = torch.amp.GradScaler("cuda")
    return scheduler, model, optimizer, ema, scaler


def load(device, file, ema_decay, num_time_steps, lr):
    scheduler, model, optimizer, ema, scaler = create(device, ema_decay, num_time_steps, lr)
    try:
        if file is not None:
            checkpoint = torch.load(file, map_location=device)
            model.load_state_dict(checkpoint["weights"])
            ema.load_state_dict(checkpoint["ema"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
    except:
        print("Could not load checkpoint")

    return scheduler, model, optimizer, ema, scaler
