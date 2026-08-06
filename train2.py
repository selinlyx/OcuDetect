import os
import json
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from models.baseline import BaselineModel
from models.ocudetect_v1 import OcuDetect
from models.ocudetect_v2 import OcuDetect2

# =====================================================================
# CONFIG - for hyperparameter tuning and ablation study
# =====================================================================
CONFIG = {
    "run_name": "model_parameters", 

    # --- hyperparameters ---
    "num_epochs": 40,
    "batch_size": 32,
    "lr": 0.001,
    "random_seed": 42,

    # --- ablation toggles  ---
    "freeze_backbone_entirely": True,  # True = backbone forever frozen
    "unfreeze_epoch": 5,                # epoch to unfreeze, when freeze_backbone_entirely=False,
    "unfreeze_last_n_blocks": None,     # None = unfreeze the whole backbone
                                        # n = last n blocks of backbone to unfreeze
    "weight_decay": 0.0,                
    "use_scheduler": False,      
    "use_augmentation": True,
    "use_class_weighted_loss": False,  
    "use_weighted_sampler": False, 
    "use_focal_loss": False,      
    "focal_gamma": 2.0,          
    "label_smoothing": 0.1,  

    # --- architecture toggles ---
    "model_name": "ocudetect_v2",  # "baseline", "ocudetect_v1", "ocudetect_v2"
    "attn_dim": 64,    
    "dropout_rate": 0.5,
    "use_attention": True,    

    # --- fixed paths (shouldn't need to change) ---
    "image_dir": "ODIR-5K/data",
    "train_csv": "ODIR-5K/train_labels.csv",
    "val_csv": "ODIR-5K/val_labels.csv",
    "test_csv": "ODIR-5K/test_labels.csv",
    "checkpoint_dir": "checkpoints",
    "results_dir": "results",
}
# =====================================================================

IMG_SIZE = 224
CLASS_NAMES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract",
               "AMD", "Hypertensive Retinopathy", "Pathological Myopia", "Other"]
LABEL_COLUMNS = ["N", "D", "G", "C", "A", "H", "M", "O"]

MODEL_REGISTRY = {
    "baseline": BaselineModel,
    "ocudetect_v1": OcuDetect,
    "ocudetect_v2": OcuDetect2
}


class OcularDataset(Dataset):
    def __init__(self, data_csv, image_dir, image_size=(224, 224), train=False, use_augmentation=False):

        self.df = pd.read_csv(data_csv)
        self.image_dir = image_dir
        self.image_size = image_size
        self.label_columns = LABEL_COLUMNS

        # augmentation is now a config toggle: train=True + use_augmentation=False
        # gives you plain resize/normalize on the train split (row0 behaviour),
        # train=True + use_augmentation=True gives you the full augmentation
        # pipeline (row4 behaviour). Val/test never get augmented either way.
        if train and use_augmentation:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02),
                transforms.ToTensor(),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        labels = row[self.label_columns].values.astype(np.float32)

        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32)


class WeightedBCELoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, outputs, labels):
        self.class_weights = self.class_weights.to(outputs.device)
        loss = nn.functional.binary_cross_entropy(outputs, labels, reduction='none')
        weighted_loss = loss * self.class_weights.unsqueeze(0)
        return weighted_loss.mean()


class FocalLoss(nn.Module):
    '''
    Focal loss for multi-label classification (Lin et al., 2017), adapted
    for a sigmoid+BCE multi-label setup.

    Unlike WeightedBCELoss (which upweights rare CLASSES uniformly, regardless
    of whether the model already gets them right), focal loss downweights
    EASY, already-correct predictions and concentrates gradient on HARD,
    currently-wrong ones -- for each individual prediction, not just each class.
    This can help more than class weighting alone when misclassifications are
    concentrated on specific hard examples rather than spread evenly within
    a class.

    alpha: optional per-class weight tensor (same shape as WeightedBCELoss's
           class_weights). Pass this in if you want focal loss AND class
           balancing at the same time. Leave as None to test focal loss on
           its own, with no class weighting.
    gamma: focusing parameter. gamma=0 reduces focal loss to plain BCE.
           Typical values are 1-5; 2.0 is the standard default from the paper.
    '''
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, labels):
        # standard BCE per-element, not yet reduced
        bce = nn.functional.binary_cross_entropy(outputs, labels, reduction='none')

        # pt = model's estimated probability of the TRUE class for each
        # element (close to 1 when the model is already confidently correct,
        # close to 0 when it's confidently wrong)
        pt = torch.exp(-bce)

        # (1 - pt)^gamma shrinks the loss contribution of easy/correct
        # elements (pt near 1) much faster than hard/wrong ones (pt near 0)
        focal = (1 - pt) ** self.gamma * bce

        if self.alpha is not None:
            alpha = self.alpha.to(outputs.device)
            focal = focal * alpha.unsqueeze(0)

        return focal.mean()


class LabelSmoothingWrapper(nn.Module):
    '''
    Wraps ANY of the three loss functions above (plain BCE, WeightedBCELoss,
    or FocalLoss) and softens the hard 0/1 targets before the wrapped loss
    sees them:
        label 1 -> (1 - smoothing)
        label 0 -> smoothing
    e.g. smoothing=0.1 turns a "1" target into 0.9 and a "0" target into 0.1.

    Why: BCE (in any of its three variants) pushes the model to drive its
    output probability all the way to the hard target. Over many epochs this
    can make the model increasingly overconfident (outputs pushed toward 0
    or 1) even on samples it's not actually more correct on -- this shows up
    as val LOSS climbing while val ERROR stays flat, since loss is sensitive
    to confidence/calibration but error only cares which side of the
    threshold the prediction landed on. Smoothing the targets caps how
    extreme the model is rewarded for being, directly countering that
    specific failure mode.

    smoothing=0.0 (the default) makes this an exact no-op passthrough to the
    wrapped criterion -- identical behaviour to not using this wrapper at all.
    '''
    def __init__(self, base_criterion, smoothing=0.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.smoothing = smoothing

    def forward(self, outputs, labels):
        if self.smoothing > 0:
            labels = labels * (1 - 2 * self.smoothing) + self.smoothing
        return self.base_criterion(outputs, labels)


def compute_sample_weights(df, label_columns):
    class_counts = df[label_columns].sum()
    class_weights = 1.0 / np.sqrt(class_counts.replace(0, 1))

    sample_weights = df[label_columns].apply(
        lambda row: max(class_weights[col] for col in label_columns if row[col] == 1)
        if row[label_columns].sum() > 0 else class_weights.min(),
        axis=1
    )
    return torch.DoubleTensor(sample_weights.values)


def compute_class_weights(train_csv, label_columns=LABEL_COLUMNS):
    df = pd.read_csv(train_csv)
    counts = df[label_columns].sum().values
    n_total = len(df)
    num_classes = len(label_columns)
    weights = n_total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_threshold_tuning(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_probs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.vstack(all_probs), np.vstack(all_labels)


def tune_thresholds_grid(val_probs, val_labels, class_names, thresholds=np.arange(0.05, 0.96, 0.05)):
    best_thresholds, best_f1s = {}, {}
    for i, name in enumerate(class_names):
        best_f1, best_t = -1, 0.5
        for t in thresholds:
            preds = (val_probs[:, i] > t).astype(int)
            f1 = f1_score(val_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[name] = float(best_t)
        best_f1s[name] = float(best_f1)

    print("\n" + "=" * 50)
    print("TUNED THRESHOLDS (on validation set)")
    print("=" * 50)
    for name in class_names:
        print(f"  {name:<25} threshold={best_thresholds[name]:.2f}  (val F1={best_f1s[name]:.4f})")
    return best_thresholds


def save_thresholds(thresholds, run_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{run_name}_thresholds.csv")
    df = pd.DataFrame(list(thresholds.items()), columns=["Disease", "Threshold"])
    df.to_csv(path, index=False)
    print(f"Thresholds saved to: {path}")
    return path


def get_data_loader(cfg):
    train_dataset = OcularDataset(cfg["train_csv"], cfg["image_dir"], (IMG_SIZE, IMG_SIZE),
                                   train=True, use_augmentation=cfg["use_augmentation"])
    val_dataset = OcularDataset(cfg["val_csv"], cfg["image_dir"], (IMG_SIZE, IMG_SIZE), train=False)
    test_dataset = OcularDataset(cfg["test_csv"], cfg["image_dir"], (IMG_SIZE, IMG_SIZE), train=False)

    if cfg["use_weighted_sampler"]:
        sample_weights = compute_sample_weights(train_dataset.df, LABEL_COLUMNS)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    print(f"\nDataset Statistics:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")

    return train_loader, val_loader, test_loader


def evaluate(model, dataloader, device, criterion, threshold=0.5):
    model.eval()
    total_loss, total_err, total_samples = 0.0, 0.0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)

            predictions = (outputs > threshold).float()
            total_err += (predictions != labels).float().sum().item()
            total_samples += labels.numel()

            all_probs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    err = float(total_err) / total_samples
    loss = total_loss / total_samples

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    all_preds = (all_probs > threshold).astype(int)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {'error': err, 'loss': loss, 'f1': macro_f1}


def build_criterion(cfg):
    # class_weights are computed the same way regardless of which loss uses
    # them -- WeightedBCELoss uses them as a flat multiplier, FocalLoss uses
    # them as its alpha term. Only computed if actually needed.
    class_weights = compute_class_weights(cfg["train_csv"]) if cfg["use_class_weighted_loss"] else None

    if cfg["use_focal_loss"]:
        # focal loss, optionally combined with class weights as alpha
        # (use_class_weighted_loss=True + use_focal_loss=True -> focal+weighted;
        #  use_class_weighted_loss=False + use_focal_loss=True -> focal alone)
        base_criterion = FocalLoss(alpha=class_weights, gamma=cfg["focal_gamma"])
    elif cfg["use_class_weighted_loss"]:
        base_criterion = WeightedBCELoss(class_weights)
    else:
        base_criterion = nn.BCELoss()

    # label smoothing wraps whichever base criterion was picked above --
    # cfg["label_smoothing"]=0.0 (default) makes this an exact no-op, so
    # existing runs/configs are completely unaffected unless you explicitly
    # set label_smoothing > 0.
    return LabelSmoothingWrapper(base_criterion, smoothing=cfg["label_smoothing"])


def build_optimizer(model, cfg, backbone_frozen):
    # while frozen, only classifier(+attention) params require grad anyway,
    # so a single param group is fine. weight_decay=0.0 reproduces plain Adam
    # with no decay if that's what the config row wants.
    return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def unfreeze_backbone_params(model, cfg):
    '''
    Unfreezes the backbone according to cfg["unfreeze_last_n_blocks"]:
    - None -> unfreezes ALL backbone params (old/original behaviour)
    - integer N -> unfreezes only the LAST N top-level children of
      model.backbone.features, leaving everything before that frozen.

    For EfficientNet-B0, model.backbone.features is a Sequential of 9
    top-level blocks (stem conv + 7 MBConv stages + final 1x1 conv).
    unfreeze_last_n_blocks=2 unfreezes roughly the last MBConv stage +
    final conv -- the highest-level, most task-specific features -- while
    keeping the early generic-feature blocks frozen, which is what actually
    limits how many parameters can overfit during fine-tuning.

    Returns the list of parameters that got unfrozen (used to build the
    optimizer's backbone param group -- frozen params should NOT be passed
    to the optimizer at all, or they'll sit in the param group with
    requires_grad=False and just waste memory/compute).
    '''
    n = cfg["unfreeze_last_n_blocks"]

    if n is None:
        # original behaviour: unfreeze everything
        for param in model.backbone.parameters():
            param.requires_grad = True
        return list(model.backbone.parameters())

    # partial unfreeze -- only works for the ocudetect_v1/EfficientNet-B0
    # backbone shape (model.backbone.features is a Sequential). If you're
    # running model_name="baseline" (ResNet-50), leave unfreeze_last_n_blocks
    # as None -- ResNet-50's structure doesn't match this block layout.
    blocks = list(model.backbone.features.children())
    n = min(n, len(blocks))
    blocks_to_unfreeze = blocks[-n:]

    unfrozen_params = []
    for block in blocks_to_unfreeze:
        for param in block.parameters():
            param.requires_grad = True
            unfrozen_params.append(param)

    print(f"  Partial unfreeze: last {n} of {len(blocks)} backbone blocks "
          f"({sum(p.numel() for p in unfrozen_params):,} params)")
    return unfrozen_params


def build_unfreeze_optimizer(model, cfg, unfrozen_backbone_params=None):
    # if partial unfreezing was used, unfrozen_backbone_params is the exact
    # (smaller) list of params that got requires_grad=True. If None (full
    # unfreeze, cfg["unfreeze_last_n_blocks"]=None), falls back to ALL
    # backbone params, matching the original behaviour exactly.
    backbone_params = unfrozen_backbone_params if unfrozen_backbone_params is not None \
        else list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    param_groups = [{'params': backbone_params, 'lr': cfg["lr"] / 10}]
    if other_params:
        param_groups.append({'params': other_params, 'lr': cfg["lr"]})
    return torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])


def maybe_build_scheduler(optimizer, cfg):
    if cfg["use_scheduler"]:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    return None


def train(model, train_loader, val_loader, device, cfg):

    model = model.to(device)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg, backbone_frozen=True)
    scheduler = maybe_build_scheduler(optimizer, cfg)

    num_epochs = cfg["num_epochs"]
    threshold = 0.5

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    train_f1 = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    val_f1 = np.zeros(num_epochs)

    best_val_err = float('inf')
    best_val_loss = float('inf')
    best_val_f1 = -1.0
    best_epoch = 0

    for epoch in range(num_epochs):
        # freeze_backbone_entirely overrides unfreeze_epoch -- if True, this
        # branch never fires, regardless of what unfreeze_epoch is set to.
        # This is what keeps model_name="baseline" runs faithful to how the
        # ResNet-50 baseline was actually trained (frozen the whole time),
        # even though the SAME loop is also used for OcuDetect runs where
        # unfreezing partway through is exactly what you want.
        if (not cfg["freeze_backbone_entirely"]) and epoch == cfg["unfreeze_epoch"]:
            print(f"UNFREEZING BACKBONE at epoch {epoch + 1}")
            unfrozen_params = unfreeze_backbone_params(model, cfg)
            optimizer = build_unfreeze_optimizer(model, cfg, unfrozen_backbone_params=unfrozen_params)
            scheduler = maybe_build_scheduler(optimizer, cfg)

        model.train()
        total_train_err, total_train_loss, total_samples = 0.0, 0.0, 0
        train_probs_list, train_labels_list = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predictions = (outputs > threshold).float()
            total_train_err += (predictions != labels).float().sum().item()
            total_train_loss += loss.item() * len(labels)
            total_samples += labels.numel()

            train_probs_list.append(outputs.detach().cpu().numpy())
            train_labels_list.append(labels.cpu().numpy())

        train_err[epoch] = float(total_train_err) / total_samples
        train_loss[epoch] = float(total_train_loss) / total_samples
        train_probs = np.vstack(train_probs_list)
        train_labels_arr = np.vstack(train_labels_list)
        train_preds = (train_probs > threshold).astype(int)
        train_f1[epoch] = f1_score(train_labels_arr, train_preds, average='macro', zero_division=0)

        val_metrics = evaluate(model, val_loader, device, criterion)
        val_err[epoch] = val_metrics['error']
        val_loss[epoch] = val_metrics['loss']
        val_f1[epoch] = val_metrics['f1']

        if scheduler is not None:
            scheduler.step(val_err[epoch])
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = cfg["lr"]

        print(f"Epoch {epoch + 1:>3}/{num_epochs} | "
              f"Train err: {train_err[epoch]:.4f}, loss: {train_loss[epoch]:.4f}, F1: {train_f1[epoch]:.4f} | "
              f"Val err: {val_err[epoch]:.4f}, loss: {val_loss[epoch]:.4f}, F1: {val_f1[epoch]:.4f} | "
              f"LR: {current_lr:.2e}")

        # CHECKPOINTING: keyed on val F1, not val_err. F1 is your actual
        # target metric -- val_err is still tracked/logged for reference/
        # diagnostics only, and does not decide which epoch gets saved.
        if val_f1[epoch] > best_val_f1:
            best_val_f1 = val_f1[epoch]
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                       os.path.join(cfg["checkpoint_dir"], f'{cfg["run_name"]}_current_best.pt'))

        # val_err/val_loss are still tracked as diagnostics (e.g. to
        # sanity-check the model isn't collapsing to all-negative), just not
        # used for checkpointing.
        if val_err[epoch] < best_val_err:
            best_val_err = val_err[epoch]

        if val_loss[epoch] < best_val_loss:
            best_val_loss = val_loss[epoch]

    # save all curves
    run_name = cfg["run_name"]
    for arr, tag in [(train_err, "train_err"), (train_loss, "train_loss"), (train_f1, "train_f1"),
                     (val_err, "val_err"), (val_loss, "val_loss"), (val_f1, "val_f1")]:
        np.savetxt(os.path.join(cfg["results_dir"], f'{run_name}_{tag}.csv'), arr)

    best_path = os.path.join(cfg["checkpoint_dir"], f'{run_name}_current_best.pt')
    rename_path = os.path.join(cfg["checkpoint_dir"], f'{run_name}_epoch{best_epoch}.pt')
    os.rename(best_path, rename_path)

    # log the config + summary result together -- this is what keeps the
    # ablation table honest. Append each run as one line to a master log.
    summary = {
        "run_name": run_name,
        "best_epoch_by_val_f1": best_epoch,
        "best_val_err": float(best_val_err),
        "best_val_f1": float(best_val_f1),
        "final_val_err": float(val_err[-1]),
        "final_val_f1": float(val_f1[-1]),
        "config": cfg,
    }
    log_path = os.path.join(cfg["results_dir"], "ablation_log.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"\nRun summary appended to: {log_path}")
    print(f"  best_val_err={best_val_err:.4f} (epoch {best_epoch}) | best_val_f1={best_val_f1:.4f}")

    print("Training is complete.")
    return model, train_err, train_loss, train_f1, val_err, val_loss, val_f1, rename_path


def plot_training_curve(train_err, val_err, train_loss, val_loss, train_f1, val_f1, run_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(16, 4))
    n = len(train_err)
    epochs = range(1, n + 1)

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_err, label="Train")
    plt.plot(epochs, val_err, label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Error"); plt.title("Train vs Validation Error")
    plt.legend(loc='best'); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train vs Validation Loss")
    plt.legend(loc='best'); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_f1, label="Train")
    plt.plot(epochs, val_f1, label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Macro F1"); plt.title("Train vs Validation Macro F1")
    plt.legend(loc='best'); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{run_name}_curves.png')
    plt.savefig(save_path)
    plt.show()
    print(f"Training curves saved to: {save_path}")


def run_training(cfg=CONFIG):
    cfg = copy.deepcopy(cfg)  # never mutate the module-level CONFIG accidentally

    np.random.seed(cfg["random_seed"])
    torch.manual_seed(cfg["random_seed"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using: {device}")
    print(f"\n{'=' * 60}\nRUN: {cfg['run_name']}\n{'=' * 60}")
    print(json.dumps(cfg, indent=2))

    train_loader, val_loader, test_loader = get_data_loader(cfg)

    model_cls = MODEL_REGISTRY[cfg["model_name"]]
    model_kwargs = {"num_classes": 8, "freeze_backbone": True}
    if cfg["model_name"] == "ocudetect_v1":
        model_kwargs.update({
            "attn_dim": cfg["attn_dim"],
            "dropout_rate": cfg["dropout_rate"],
            "use_attention": cfg["use_attention"],
        })
    model = model_cls(**model_kwargs)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model.name}")
    print(f"  Total parameters: {total_params}")
    print(f"  Trainable parameters: {trainable_params} ({trainable_params/total_params*100:.1f}%)")

    # explicit confirmation so it's always visible in the console output
    # which freeze mode a run actually used -- prevents silently getting a
    # partially-unfrozen backbone when you meant to reproduce a frozen baseline
    if cfg["freeze_backbone_entirely"]:
        print(f"  Backbone: FROZEN for all {cfg['num_epochs']} epochs (freeze_backbone_entirely=True)")
    else:
        print(f"  Backbone: frozen until epoch {cfg['unfreeze_epoch'] + 1}, then unfrozen")

    # gentle nudge, not a hard stop -- catches the specific mistake of running
    # the ResNet-50 baseline with the backbone unfreezing partway through,
    # which would NOT match your originally reported baseline numbers
    if cfg["model_name"] == "baseline" and not cfg["freeze_backbone_entirely"]:
        print("  [WARNING] model_name='baseline' but freeze_backbone_entirely=False -- "
              "this will NOT reproduce your original reported baseline (which was "
              "frozen the whole time). Set freeze_backbone_entirely=True if that's "
              "what you intended, unless you're deliberately testing an unfrozen baseline.")

    model, train_err, train_loss, train_f1, val_err, val_loss, val_f1, best_model_path = train(
        model=model, train_loader=train_loader, val_loader=val_loader, device=device, cfg=cfg)

    plot_training_curve(train_err, val_err, train_loss, val_loss, train_f1, val_f1,
                         run_name=cfg["run_name"], save_dir=cfg["results_dir"])

    print("\n" + "=" * 60)
    print("STARTING THRESHOLD TUNING")
    print("=" * 60)

    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)

    val_probs, val_labels = evaluate_threshold_tuning(model, val_loader, device)
    thresholds = tune_thresholds_grid(val_probs, val_labels, CLASS_NAMES)
    save_thresholds(thresholds, cfg["run_name"], cfg["results_dir"])

    return model, best_model_path


if __name__ == "__main__":
    run_training(CONFIG)