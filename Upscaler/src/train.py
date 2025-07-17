import random
import numpy as np
import torch
from math import isnan
from torch import nn, tensor, masked_select
from torch.utils.data import DataLoader
from torch.optim import Adam
from ddpm_scheduler import DDPM_Scheduler
from unet import UNET
from timm.utils import ModelEmaV3
from tqdm import tqdm
from torch.nn.functional import pad, interpolate

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(dataset,
          batch_size: int=1,
          num_time_steps: int=1000,
          num_epochs: int=15,
          seed: int=-1,
          ema_decay: float=0.9999,  
          lr=2e-5,
          checkpoint_path: str=None):
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, prefetch_factor=2)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = UNET()
    optimizer = Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')

    for i in range(num_epochs):
        total_loss = 0
        entries = tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")
        for bidx, datapoint in enumerate(entries):
            for_train = datapoint['without_nan'] 
            mask = datapoint['mask']
            t = torch.randint(0,num_time_steps,(batch_size,))
            e = torch.randn_like(for_train, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size,1,1,1)
            for_train = (torch.sqrt(a)*for_train) + (torch.sqrt(1-a)*e)
            output = model(for_train, t)
            optimizer.zero_grad()
            loss = criterion(output, e) 
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
        print(f'Epoch {i+1} | Loss {total_loss / (60000/batch_size):.5f}')

    print("Assembled checkpoint")
    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }

    print("Saving checkpoint")
    torch.save(checkpoint, 'checkpoints/ddpm_checkpoint')
    print("Saved checkpoint")
