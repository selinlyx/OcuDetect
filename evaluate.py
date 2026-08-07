import os
# must be set before numpy/torch are imported -- their BLAS backends read
# these at init time to size their thread pools. Left unset, OpenBLAS can
# spawn more threads than available memory can back, causing:
# "OpenBLAS error: Memory allocation still failed after 10 retries"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import importlib
from train2 import OcularDataset
import pandas as pd

MODEL_KWARGS = {}

# # progress report baseline model checkpoint
# MODEL_CLASS = "BaselineModel"
# MODULE_PATH = "models.baseline"
# CHECKPOINT_FILE = "Baseline_epochs19_bs32_lr0.001.pt"

# # progress report ocudetect model (v1) checkpoint
# MODEL_CLASS = "OcuDetect"
# MODULE_PATH = "models.ocudetect_v1" 
# CHECKPOINT_FILE = "OcuDetect_v1_epochs16_bs32_lr0.001.pt"

# # final report ocudetect model (v1) trained train2.py with more aumentation and weighted sampling
# MODEL_CLASS = "OcuDetect"
# MODULE_PATH = "models.ocudetect_v1" 
# CHECKPOINT_FILE = "OcuDetect_v1_epochs20_bs32_lr0.001.pt"


# # BEST MODEL SO FAR
# final report ocudetect model (v1) trained train2.py with more aumentation and inverse sqrt weighted sampling
MODEL_CLASS = "OcuDetect"
MODULE_PATH = "models.ocudetect_v1" 
CHECKPOINT_FILE = "OcuDetect_v1_epochs17_bs32_lr0.001.pt"


# # final report ocudetect model (v1) trained train2.py ablation none
# MODEL_CLASS = "OcuDetectAblation"
# MODULE_PATH = "ablation" 
# CHECKPOINT_FILE = "OcuDetect_ablation_none_epochs15_bs32_lr0.001.pt"

# # final report ocudetect model (v1) trained train2.py ablation self_attention
# MODEL_CLASS = "OcuDetectAblation"
# MODULE_PATH = "ablation"
# CHECKPOINT_FILE = "OcuDetect_ablation_self_attention_epochs17_bs32_lr0.001.pt"
# MODEL_KWARGS = {"attention_type": "self_attention"}

# MODEL_CLASS = "OcuDetect2"
# MODULE_PATH = "models.ocudetect_v2"
# CHECKPOINT_FILE = "efficientnetb3_frozen_labelsmoothing_ocudetectv2_attn64_aug_schedule_sampler_epoch40.pt"
# MODEL_KWARGS = {"attn_dim": 64}

# MODEL_CLASS = "OcuDetect"
# MODULE_PATH = "models.ocudetect_v1"
# CHECKPOINT_FILE ="efficientnetb0_frozen_labelsmoothing_ocudetectv1_attn256_aug_schedule_sampler_epoch38.pt"
# MODEL_KWARGS = {"attn_dim": 256}

IMAGE_DIR = "ODIR-5K/data"
TEST_CSV = "ODIR-5K/test_labels.csv"
# IMAGE_DIR = "combined_test_images"
# TEST_CSV = "combined_test_labels.csv"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "evaluation_results"
THRESHOLDS_DIR = "thresholds"

CLASS_NAMES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", 
               "AMD", "Hypertensive Retinopathy", "Pathological Myopia", "Other"]
CLASS_SHORT = ["N", "D", "G", "C", "A", "H", "M", "O"]
RANDOM_SEED = 42
BATCH_SIZE = 32
THRESHOLD = 0.5

# sentinel used by preprocess_rfmid.py / remap_rfmid_labels.py to mark
# "no ground truth for this class" (e.g. RFMiD has no G/C labels at all).
# These cells must be masked out of loss and metrics, NOT treated as 0.
MASK_VALUE = -1


def load_checkpoint(checkpoint_file, model_class, module_path, model_kwargs=None):

    model_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)
    print(f"Loading: {model_path}")

    module = importlib.import_module(module_path)
    get_model_class = getattr(module, model_class)

    model = get_model_class(**(model_kwargs or {})) # create model object

    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    return model


def load_thresholds(checkpoint_file, thresholds_dir=THRESHOLDS_DIR):
    # tuned threshold files are saved under the run name, without the "_epochN" suffix
    base_name = re.sub(r'_epoch\d+$', '', os.path.splitext(checkpoint_file)[0])
    path = os.path.join(thresholds_dir, f"{base_name}_thresholds.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return dict(zip(df["Disease"], df["Threshold"]))


def apply_thresholds(all_probabilities, class_names, thresholds_dict):
    preds = np.zeros_like(all_probabilities, dtype=int)
    for i, name in enumerate(class_names):
        preds[:, i] = (all_probabilities[:, i] > thresholds_dict[name]).astype(int)
    return preds


def evaluate_model(model, test_loader, device, threshold):
    '''
    Masked evaluation: any label cell equal to MASK_VALUE (-1) is excluded
    from both the loss and the error calculation -- it means "no ground truth for
    this class on this image" (e.g. RFMiD rows have no G/C labels), NOT
    "confirmed negative". Raw labels (with -1 intact) are still returned
    in all_labels so downstream per-class metric functions can mask too.
    '''
    model.eval()
    all_predictions, all_labels, all_probabilities = [], [], []
    total_loss, total_err, total_samples = 0.0, 0.0, 0
    criterion = nn.BCELoss(reduction='none')

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            valid_mask = (labels != MASK_VALUE).float()
            # dummy value (0) for masked cells -- doesn't matter, gets zeroed by valid_mask
            safe_labels = torch.where(labels == MASK_VALUE, torch.zeros_like(labels), labels)

            loss_per_elem = criterion(outputs, safe_labels)
            masked_loss = (loss_per_elem * valid_mask).sum()
            total_loss += masked_loss.item()

            preds = (outputs > threshold).float()
            wrong = (preds != safe_labels).float() * valid_mask
            total_err += wrong.sum().item()
            total_samples += valid_mask.sum().item()

            all_predictions.extend(preds.cpu().numpy().astype(int))
            all_labels.extend(labels.cpu().numpy())  # keep -1 intact for downstream masking
            all_probabilities.extend(outputs.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    error = total_err / total_samples
    loss = total_loss / total_samples

    metrics = {
        'error': error,
        'loss': loss,
    }

    return metrics, all_predictions, all_labels, all_probabilities


def evaluate_model_with_thresholds(model, test_loader, device, thresholds_dict, class_names):
    model.eval()
    all_labels, all_probabilities = [], []
    total_loss, total_samples = 0.0, 0
    criterion = nn.BCELoss(reduction='none')

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            valid_mask = (labels != MASK_VALUE).float()
            safe_labels = torch.where(labels == MASK_VALUE, torch.zeros_like(labels), labels)

            loss_per_elem = criterion(outputs, safe_labels)
            total_loss += (loss_per_elem * valid_mask).sum().item()
            total_samples += valid_mask.sum().item()

            all_labels.extend(labels.cpu().numpy())  # keep -1 intact
            all_probabilities.extend(outputs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    all_predictions = apply_thresholds(all_probabilities, class_names, thresholds_dict)

    valid = (all_labels != MASK_VALUE)
    error = ((all_predictions != all_labels) & valid).sum() / valid.sum()
    loss = total_loss / total_samples

    metrics = {'error': error, 'loss': loss}
    return metrics, all_predictions, all_labels, all_probabilities


def print_per_class_metrics(all_labels, all_preds, class_names):
    '''
    Every per-class computation restricts to rows where that class's
    label != MASK_VALUE. Micro and weighted averages are computed
    manually from the per-class TP/FP/FN (rather than via sklearn's
    built-in multi-label aggregation), since sklearn has no concept of
    per-column masking and would otherwise silently include masked
    cells if they were coerced to 0/1 beforehand.
    '''
    rows = []

    for i, disease in enumerate(class_names):
        valid = all_labels[:, i] != MASK_VALUE
        n_masked = (~valid).sum()

        labels_i = all_labels[valid, i]
        preds_i = all_preds[valid, i]

        tp = ((preds_i == 1) & (labels_i == 1)).sum()
        fp = ((preds_i == 1) & (labels_i == 0)).sum()
        fn = ((preds_i == 0) & (labels_i == 1)).sum()
        tn = ((preds_i == 0) & (labels_i == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = (labels_i == 1).sum()

        rows.append({
            'Disease': disease,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Support': support,
            'Masked': n_masked,
            '_tp': tp, '_fp': fp, '_fn': fn, '_tn': tn,
        })

    df = pd.DataFrame(rows)

    # macro: unweighted mean across classes (this is your primary metric)
    macro_precision = np.mean(df['Precision'])
    macro_recall = np.mean(df['Recall'])
    macro_f1 = np.mean(df['F1'])

    # micro: aggregate TP/FP/FN across all classes, then compute once
    total_tp, total_fp, total_fn = df['_tp'].sum(), df['_fp'].sum(), df['_fn'].sum()
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0)

    # weighted: mean of per-class metrics weighted by each class's support
    total_support = df['Support'].sum()
    if total_support > 0:
        weighted_precision = (df['Precision'] * df['Support']).sum() / total_support
        weighted_recall = (df['Recall'] * df['Support']).sum() / total_support
        weighted_f1 = (df['F1'] * df['Support']).sum() / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0

    display_df = df.drop(columns=['_tp', '_fp', '_fn', '_tn'])

    averages = pd.DataFrame([
        {'Disease': 'MACRO AVG', 'Precision': macro_precision, 'Recall': macro_recall, 'F1': macro_f1, 'Support': '', 'Masked': ''},
        {'Disease': 'MICRO AVG', 'Precision': micro_precision, 'Recall': micro_recall, 'F1': micro_f1, 'Support': '', 'Masked': ''},
        {'Disease': 'WEIGHTED AVG', 'Precision': weighted_precision, 'Recall': weighted_recall, 'F1': weighted_f1, 'Support': '', 'Masked': ''},
    ])

    display_df = pd.concat([display_df, averages], ignore_index=True)
    for col in ['Precision', 'Recall', 'F1']:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.floating)) else x)

    print("\n" + "="*70)
    print("PER-CLASS METRICS")
    print("(Masked = rows excluded because ground truth wasn't available for that class,")
    print(" e.g. RFMiD images have no G/C labels)")
    print("="*70)
    print(display_df.to_string(index=False))

    worst_idx = df['F1'].idxmin()
    best_idx = df['F1'].idxmax()

    print(f"\nWorst performing disease: {df.loc[worst_idx, 'Disease']} (F1: {df.loc[worst_idx, 'F1']:.4f})")
    print(f"Best performing disease: {df.loc[best_idx, 'Disease']} (F1: {df.loc[best_idx, 'F1']:.4f})")

    return {
        'macro': {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1},
        'micro': {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1},
        'weighted': {'precision': weighted_precision, 'recall': weighted_recall, 'f1': weighted_f1}
    }


def find_multilabel_samples(all_predictions, all_labels, min_positives=2, n=4):
    '''
    Find up to n multilabel samples -- ground truth has >= min_positives
    positive labels (excluding masked cells) -- ranked with the most
    correctly predicted samples first. Falls back to the most
    multilabel-heavy samples available if nothing meets min_positives.
    '''
    n_positives = (all_labels == 1).sum(axis=1)
    candidates = np.where(n_positives >= min_positives)[0]

    if len(candidates) == 0:
        # relax to the most multilabel samples the test set actually has
        max_positives = n_positives.max()
        if max_positives < 2:
            raise ValueError("No multilabel samples found in this test set")
        candidates = np.where(n_positives == max_positives)[0]

    scored = []
    for idx in candidates:
        valid = all_labels[idx] != MASK_VALUE
        n_valid = valid.sum()
        score = (all_predictions[idx][valid] == all_labels[idx][valid]).sum() / n_valid if n_valid > 0 else 0
        scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in scored[:n]]


def build_sample_text(idx, prediction, truth, probability):
    is_correct = np.array_equal(prediction, truth)
    lines = []
    lines.append(f"Sample {idx} | {'CORRECT' if is_correct else 'INCORRECT'}")
    lines.append("")
    lines.append("Disease   | GT | Pred | Prob")
    lines.append("-" * 35)

    for j, name in enumerate(CLASS_NAMES):
        gt_display = "?" if truth[j] == MASK_VALUE else str(truth[j])
        match = "--" if truth[j] == MASK_VALUE else ("OK" if prediction[j] == truth[j] else "XX")
        lines.append(f"{CLASS_SHORT[j]:<9} | {gt_display:<2} | {prediction[j]:<4} | {probability[j]:.3f} {match}")

    lines.append("")
    lines.append("GT:  " + ", ".join([CLASS_SHORT[j] for j, v in enumerate(truth) if v == 1]) or "None")
    lines.append("Pred: " + ", ".join([f"{CLASS_SHORT[j]}({probability[j]:.2f})" for j, v in enumerate(prediction) if v == 1]) or "None")

    return "\n".join(lines)


def display_results(metrics, checkpoint_name, all_predictions, all_labels, all_probabilities, test_dataset, n_samples=3):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"EVALUATION RESULTS - {checkpoint_name}")
    print(f"\nOverall Metrics:")
    print(f"  Error: {metrics['error']:.4f} ({metrics['error']*100:.2f}%)")
    print(f"  Loss: {metrics['loss']:.4f}")

    indices = find_multilabel_samples(all_predictions, all_labels, min_positives=2, n=n_samples)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    n = len(indices)
    # images across the top, each sample's text directly below its own image
    # (same column) rather than in a separate wide column beside it
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 7.5),
                              gridspec_kw={'height_ratios': [1, 0.75]})
    if n == 1:
        axes = axes[:, np.newaxis]

    for col, idx in enumerate(indices):
        img, _ = test_dataset[idx]
        prediction = all_predictions[idx]
        truth = all_labels[idx]
        probability = all_probabilities[idx]

        # convert tensor to displayable format
        img = img.numpy().transpose(1, 2, 0)
        img = np.clip(img * std + mean, 0, 1)

        ax_img, ax_text = axes[0, col], axes[1, col]

        ax_img.imshow(img)
        ax_img.axis('off')

        # text contained within this sample's own column, top-aligned under its image
        ax_text.axis('off')
        ax_text.text(
            0.5, 1.0,
            build_sample_text(idx, prediction, truth, probability),
            transform=ax_text.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='center',
            family='monospace'
        )

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sample_predictions.png'), bbox_inches='tight')
    plt.show()

    print(f"\nImage saved to: {RESULTS_DIR}/sample_predictions.png")


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


def compute_and_plot_confusion_matrix(all_labels, all_preds, class_names, save_dir=RESULTS_DIR):
    """
    Compute and plot confusion matrices with compact cells and big text.
    Each per-class matrix is computed only over rows where that class's
    label != MASK_VALUE. The "overall" flattened matrix at the bottom
    also excludes every masked cell before flattening.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_classes = len(class_names)
    metrics_summary = {}
    
    # ===== INDIVIDUAL CONFUSION MATRICES =====
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))  # Slightly wider for titles
    axes = axes.flatten()
    
    for i, name in enumerate(class_names):
        valid = all_labels[:, i] != MASK_VALUE
        labels_i = all_labels[valid, i]
        preds_i = all_preds[valid, i]

        cm = confusion_matrix(labels_i, preds_i, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_summary[name] = {
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Tight heatmap with big numbers
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['N', 'Y'], yticklabels=['N', 'Y'],
                   cbar=False,
                   annot_kws={'size': 22},
                   square=True)
        
        # ===== BIGGER DISEASE NAME =====
        axes[i].set_title(f'{name}', fontsize=14, fontweight='bold')  # ← 12 → 14
        
        # Big P/R labels
        axes[i].set_xlabel(f'P:{precision:.2f} R:{recall:.2f}', fontsize=13)
        axes[i].set_ylabel('True', fontsize=11)
        axes[i].tick_params(labelsize=10)
        
        # Make numbers bold
        for text in axes[i].texts:
            text.set_fontsize(22)
            text.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_individual.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===== OVERALL CONFUSION MATRIX =====
    overall_valid = (all_labels != MASK_VALUE)
    all_preds_flat = all_preds[overall_valid]
    all_labels_flat = all_labels[overall_valid]
    cm_overall = confusion_matrix(all_labels_flat, all_preds_flat, labels=[0, 1])
    
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                annot_kws={'size': 22},
                square=True)
    ax.set_title('Overall Confusion Matrix', fontsize=14, fontweight='bold')  # ← Bigger
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_overall.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===== PRINT SUMMARY TABLE =====
    print("\n" + "="*70)
    print("CONFUSION MATRIX SUMMARY")
    print("="*70)
    print(f"{'Disease':<25} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print("-"*95)
    
    for name, m in metrics_summary.items():
        print(f"{name:<25} {m['TP']:<6} {m['FP']:<6} {m['FN']:<6} {m['TN']:<6} "
              f"{m['precision']:.4f}  {m['recall']:.4f}  {m['f1']:.4f}")
    
    print("-"*95)
    
    best_disease = max(metrics_summary.items(), key=lambda x: x[1]['f1'])
    worst_disease = min(metrics_summary.items(), key=lambda x: x[1]['f1'])
    
    print(f"\nBest: {best_disease[0]} (F1: {best_disease[1]['f1']:.4f})")
    print(f"Worst: {worst_disease[0]} (F1: {worst_disease[1]['f1']:.4f})")
    
    return metrics_summary


def run_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using: {device}")
    
    # load test data
    test_dataset = OcularDataset(TEST_CSV, IMAGE_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    
    # load model
    model = load_checkpoint(CHECKPOINT_FILE, MODEL_CLASS, MODULE_PATH, MODEL_KWARGS)
    model = model.to(device)

    # load thresholds tuned in train2.py, if available for this checkpoint
    thresholds = load_thresholds(CHECKPOINT_FILE)

    if thresholds is not None:
        print(f"Using tuned thresholds: {thresholds}")
        metrics, all_preds, all_labels, all_probs = evaluate_model_with_thresholds(model, test_loader, device, thresholds, CLASS_NAMES)
    else:
        print(f"No tuned thresholds found for {CHECKPOINT_FILE}, using fixed threshold: {THRESHOLD}")
        metrics, all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device, THRESHOLD)

    # display confusion matrix metrics
    compute_and_plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES, RESULTS_DIR)

    print_per_class_metrics(all_labels, all_preds, CLASS_NAMES)

    # display a sample result
    display_results(metrics, CHECKPOINT_FILE, all_preds, all_labels, all_probs, test_dataset)

if __name__ == "__main__":
    run_evaluate()