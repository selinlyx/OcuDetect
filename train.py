import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import roc_auc_score, f1_score
from torchvision import transforms
from PIL import Image
from models.baseline import BaselineModel
import matplotlib.pyplot as plt


# config =========================================================
IMG_SIZE = 224 
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 0.001
IMAGE_DIR = "ODIR-5K/data" 
TRAIN_CSV = "ODIR-5K/train_labels.csv"
VAL_CSV = "ODIR-5K/val_labels.csv"
TEST_CSV = "ODIR-5K/test_labels.csv"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR =  "results"
RANDOM_SEED = 42
# ================================================================

class OcularDataset(Dataset):
    def __init__(self, data_csv, image_dir, image_size=(224, 224)):
 
        self.df = pd.read_csv(data_csv)
        self.image_dir = image_dir
        self.image_size = image_size
        
        # labels
        self.label_columns = ["N", "D", "G", "C", "A", "H", "M", "O"]
        
        # define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),                # convert to tensor
            transforms.Normalize(                 # normalize to [0, 1]
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # get filename and labels
        row = self.df.iloc[idx]
        filename = row['filename']
        labels = row[self.label_columns].values.astype(np.float32)
        
        # load preprocessed image
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')
        
        # apply transforms
        image = self.transform(image)
        
        return image, torch.tensor(labels, dtype=torch.float32)


def get_data_loader(train_csv, val_csv, test_csv, image_dir, batch_size=32, image_size=(224, 224)):

    # create datasets
    train_dataset = OcularDataset(train_csv, image_dir, image_size)
    val_dataset = OcularDataset(val_csv, image_dir, image_size)
    test_dataset = OcularDataset(test_csv, image_dir, image_size)
    
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # print size of split datasets
    print(f"\nDataset Statistics:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader


def evaluate(model, dataloader, device, criterion, threshold=0.5):
    model.eval() # switch to eval mode

    total_loss = 0 # sum loss
    total_err = 0 # sum error
    total_samples = 0 # sum samples
    
    # all_probabilites = [] # to compute AUC
    # all_predictions = []
    # all_labels = []

    with torch.no_grad(): # no updating weights in eval
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels) # add to loss
            
            # for multi-label with sigmoid (threshold 0.5)
            predictions = (outputs > threshold).float()

            # add to error
            total_err += (predictions != labels).float().sum().item()
            total_samples += labels.numel() # count total samples

    # compute average error over total samples
    err = float(total_err) / total_samples

    # compute average loss over batches
    loss = total_loss / total_samples
    
    return {'error': err, 'loss': loss}


def train(model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS, 
          batch_size=BATCH_SIZE, learning_rate=LR, results_dir=RESULTS_DIR,
          checkpoint_dir=CHECKPOINT_DIR, threshold=0.5, unfreeze_epoch=5):

    model = model.to(device)

    os.makedirs(results_dir, exist_ok=True) # make save directory if not already existing
    os.makedirs(checkpoint_dir, exist_ok=True) # make save directory if not already existing

    criterion = nn.BCELoss() # define loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # define optimizer
    
    # define arrays to store train and validation loss and error
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs) 
    val_loss = np.zeros(num_epochs)
    
    best_val_err = float('inf') # to track best model based on error
    best_epoch = 0

    # training loops 
    for epoch in range(num_epochs):
        if epoch == unfreeze_epoch:
           
            print(f"UNFREEZING BACKBONE at epoch {epoch+1}")
            
            # unfreeze backbone
            for param in model.backbone.parameters():
                param.requires_grad = True
            
            # lower LR for backbone, higher for classifier
            optimizer = torch.optim.Adam([
                {'params': model.backbone.parameters(), 'lr': learning_rate / 10},
                {'params': model.classifier.parameters(), 'lr': learning_rate}
            ])

        model.train()

        # to accumulate error and loss 
        total_train_err = 0.0
        total_train_loss = 0.0
        total_samples = 0
        
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad() # zero parameter gradients
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() # update weights
            
            predictions = (outputs > threshold).float()
            total_train_err += (predictions != labels).float().sum().item()
            total_train_loss += loss.item() * len(labels)
            total_samples += labels.numel()
        
        train_err[epoch] = float(total_train_err) / total_samples 
        train_loss[epoch] = float(total_train_loss) / total_samples
        
        val_metrics = evaluate(model, val_loader, device, criterion)
        val_err[epoch] = val_metrics['error']
        val_loss[epoch] = val_metrics['loss']
        
        print(f"Epoch {epoch+1}: Train err: {train_err[epoch]:.4f}, Train loss: {train_loss[epoch]:.4f} |")
        print(f"  Validation err: {val_err[epoch]:.4f}, Validation loss: {val_loss[epoch]:.4f}")

        if val_err[epoch] < best_val_err:
            best_val_err = val_err[epoch]
            best_epoch = epoch + 1

            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'{model.name}_current_best_model.pt'))
    
    np.savetxt(os.path.join(results_dir, f'{model.name}_epochs{best_epoch}_bs{batch_size}_lr{learning_rate}_train_err.csv'), train_err)
    np.savetxt(os.path.join(results_dir, f'{model.name}_epochs{best_epoch}_bs{batch_size}_lr{learning_rate}_train_loss.csv'), train_loss)
    np.savetxt(os.path.join(results_dir, f'{model.name}_epochs{best_epoch}_bs{batch_size}_lr{learning_rate}_val_err.csv'), val_err)
    np.savetxt(os.path.join(results_dir, f'{model.name}_epochs{best_epoch}_bs{batch_size}_lr{learning_rate}_val_loss.csv'), val_loss)
    
    best_path = os.path.join(checkpoint_dir, f'{model.name}_current_best_model.pt')
    rename_path = os.path.join(checkpoint_dir, f'{model.name}_epochs{best_epoch}_bs{batch_size}_lr{learning_rate}.pt')
    os.rename(best_path, rename_path)
    print(f"Training is complete.")

    return model, train_err, train_loss, val_err, val_loss


def plot_training_curve(train_err, val_err, train_loss, val_loss, save_path, save_dir=RESULTS_DIR):

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 4))

    n = len(train_err)
    epochs = range(1, n + 1)

    # error vs epochs curve 
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_err, label="Train")
    plt.plot(epochs, val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Train vs Validation Error")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # loss vs epochs curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{save_path}.png'))
    plt.show()
    
    print(f"Training curves saved to: {os.path.join(save_dir, f'{save_path}.png')}")

def run_training():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # setup the device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # get data loaders 
    train_loader, val_loader, test_loader = get_data_loader(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        image_dir=IMAGE_DIR,
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE)
    )

    model = BaselineModel(num_classes=8, freeze_backbone=True) # define model object
    
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"\nModel: {model.name}")
    print(f"  Total parameters: {total_params}")
    print(f"  Trainable parameters: {trainable_params} ({trainable_params/total_params*100:.1f}%)")
    
    # train model 
    model, train_err, train_loss, val_err, val_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        results_dir=RESULTS_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        threshold=0.5
    )
    
    # plot training curves
    plot_training_curve(
        train_err=train_err,
        val_err=val_err,
        train_loss=train_loss,
        val_loss=val_loss,
        save_path=f"{model.name}_bs{BATCH_SIZE}_lr{LR}_training_curves",
        save_dir=RESULTS_DIR
    )


if __name__ == "__main__":
    run_training()