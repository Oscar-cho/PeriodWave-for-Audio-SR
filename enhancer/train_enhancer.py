import os
import sys
import torch
import torch.nn as nn
import argparse
import json
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add the parent directory to the path to find the 'utils' and 'audiosr' modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from audiosr.unet_enhancer import UNet
from audiosr.vctk_dataset import VCTKDataset

def main():
    # Re-introduce argparse for file paths
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="path to configuration file")
    parser.add_argument('-m', '--model_dir', type=str, required=True,
                        help="directory to save models")
    parser.add_argument('--vctk_path', type=str, required=True,
                        help="path to the root of the VCTK dataset")
    args = parser.parse_args()

    # This check ensures the script is run from the project root.
    if not os.path.exists('utils.py') or not os.path.exists(args.config):
        print("Error: This script must be run from the root of the PeriodWave project.")
        print("Please make sure 'utils.py' and the config file exist in your current directory.")
        return

    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config_train = config['train']
    config_data = config['data']

    # Create model directory from args
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = UNet(n_channels=1, n_out_channels=1).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_train.get('learning_rate', 1e-4))
    
    # Loss function
    criterion = nn.L1Loss().to(device)

    # Datasets and Dataloaders using the provided path
    print(f"Loading VCTK dataset from path: {args.vctk_path}")
    train_dataset = VCTKDataset(args.vctk_path, config_data, segment_size=config_data['segment_size'], train=True)
    train_loader = DataLoader(train_dataset, batch_size=config_train['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    val_dataset = VCTKDataset(args.vctk_path, config_data, segment_size=config_data['segment_size'], train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print("Dataset loading complete.")

    # Tensorboard writer
    writer = SummaryWriter(log_dir=args.model_dir)

    print("Starting training...")
    # Training loop
    model.train()
    global_step = 0
    for epoch in range(config_train.get('epochs', 20000)):
        print(f"Epoch: {epoch+1}/{config_train.get('epochs', 20000)}")
        
        # Training
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for i, (mel_lr, mel_hr) in enumerate(pbar):
            mel_lr = mel_lr.to(device)
            mel_hr = mel_hr.to(device)

            optimizer.zero_grad()

            pred_mel_hr = model(mel_lr.unsqueeze(1)).squeeze(1)
            loss = criterion(pred_mel_hr, mel_hr)

            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=loss.item())

            if global_step % 100 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
            
            global_step += 1

        # Validation
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc="Validating")
        with torch.no_grad():
            for i, (mel_lr, mel_hr) in enumerate(val_pbar):
                mel_lr = mel_lr.to(device)
                mel_hr = mel_hr.to(device)

                pred_mel_hr = model(mel_lr.unsqueeze(1)).squeeze(1)
                loss = criterion(pred_mel_hr, mel_hr)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        writer.add_scalar('val/loss', val_loss, global_step)
        print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")

        # Save checkpoint every 50 epochs or on the last epoch
        if (epoch + 1) % 50 == 0 or (epoch + 1) == config_train.get('epochs', 200):
            checkpoint_path = os.path.join(args.model_dir, f"enhancer_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        model.train()

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main()
