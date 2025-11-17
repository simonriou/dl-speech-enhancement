import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from cnn import build_cnn_mask_model
from data import SpectrogramDataset
from tqdm import tqdm

os.makedirs('./checkpoints/', exist_ok=True)
os.makedirs('./models/', exist_ok=True)
os.makedirs('./history/', exist_ok=True)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"Using device: {device}")

freq_bins, n_frames = 513, 188
batch_size = 32
epochs = 15
validation_split = 0.2

# Build model
model = build_cnn_mask_model().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Datasets & loaders
train_dataset = SpectrogramDataset(
    features_dir='./data/train/features/',
    labels_dir='./data/train/labels/',
    validation_split=validation_split,
    subset='training'
)
val_dataset = SpectrogramDataset(
    features_dir='./data/train/features/',
    labels_dir='./data/train/labels/',
    validation_split=validation_split,
    subset='validation'
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Batch size: {batch_size}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# --- Training loop ---
train_loss_history = []
val_loss_history = []

for epoch in range(epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"{'='*60}")
    
    # Training
    model.train()
    train_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (X, y) in enumerate(train_loader_tqdm):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X.size(0)
        
        # Update progress bar with current loss
        train_loader_tqdm.set_postfix({'batch_loss': f'{loss.item():.4f}'})
    
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for X_val, y_val in val_loader_tqdm:
            X_val, y_val = X_val.to(device), y_val.to(device)
            y_pred = model(X_val)
            loss = criterion(y_pred, y_val)
            val_loss += loss.item() * X_val.size(0)
            
            val_loader_tqdm.set_postfix({'batch_loss': f'{loss.item():.4f}'})
    
    val_loss /= len(val_loader.dataset)

    print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    # Show improvement
    if epoch > 0:
        train_diff = train_loss_history[-1] - train_loss
        val_diff = val_loss_history[-1] - val_loss
        print(f"Train Δ: {train_diff:+.6f} | Val Δ: {val_diff:+.6f}")

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    # Save checkpoint
    checkpoint_path = f'./checkpoints/model_epoch_{epoch+1:02d}.pt'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# Save final model
torch.save(model.state_dict(), './models/model_final.pt')

# Save training history
np.save('./history/loss.npy', np.array(train_loss_history))
np.save('./history/val_loss.npy', np.array(val_loss_history))

print("\n" + "="*60)
print("Training complete!")
print(f"Final Train Loss: {train_loss_history[-1]:.6f}")
print(f"Final Val Loss: {val_loss_history[-1]:.6f}")
print(f"Best Val Loss: {min(val_loss_history):.6f} (Epoch {np.argmin(val_loss_history)+1})")
print("="*60)