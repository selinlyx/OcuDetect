import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import  precision_score, recall_score, f1_score, confusion_matrix
import importlib
from train import OcularDataset
import pandas as pd

MODEL_CLASS = "BaselineModel"
MODULE_PATH = "models.baseline" 
CHECKPOINT_FILE = "Baseline_epochs19_bs32_lr0.001.pt"

# MODEL_CLASS = "OcuDetect"
# MODULE_PATH = "models.ocudetect_v1" 
# CHECKPOINT_FILE = "OcuDetect_v1_epochs18_bs32_lr0.001.pt"

IMAGE_DIR = "ODIR-5K/data"
TEST_CSV = "ODIR-5K/test_labels.csv"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "evaluation_results"

CLASS_NAMES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", 
               "AMD", "Hypertensive Retinopathy", "Pathological Myopia", "Other"]
CLASS_SHORT = ["N", "D", "G", "C", "A", "H", "M", "O"]
RANDOM_SEED = 42
BATCH_SIZE = 32
THRESHOLD = 0.5

def load_checkpoint(checkpoint_file, model_class, module_path):

    model_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)
    print(f"Loading: {model_path}")
    
    module = importlib.import_module(module_path)
    get_model_class = getattr(module, model_class)
    
    model = get_model_class() # create model object
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    return model


def evaluate_model(model, test_loader, device, threshold):

    model.eval()
    all_predictions, all_labels, all_probabilities = [], [], []
    total_loss, total_err, total_samples = 0.0, 0.0, 0
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            
            preds = (outputs > threshold).float()
            total_err += (preds != labels).float().sum().item()
            total_samples += labels.numel()
            
            all_predictions.extend(preds.cpu().numpy().astype(int))
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(outputs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # compute metrics
    error = total_err / total_samples
    loss = total_loss / total_samples
    
    metrics = {
        'error': error,
        'loss': loss,
    }
    
    return metrics, all_predictions, all_labels, all_probabilities

def print_per_class_metrics(all_labels, all_preds, class_names):
    
    rows = [] 
    
    for i, disease in enumerate(class_names):
        tp = ((all_preds[:, i] == 1) & (all_labels[:, i] == 1)).sum()
        fp = ((all_preds[:, i] == 1) & (all_labels[:, i] == 0)).sum()
        fn = ((all_preds[:, i] == 0) & (all_labels[:, i] == 1)).sum()
        tn = ((all_preds[:, i] == 0) & (all_labels[:, i] == 0)).sum()
        
        # compute precision, recall, f1, and support
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = (all_labels[:, i] == 1).sum()
        
        rows.append({
            'Disease': disease,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Support': support
        })

    df = pd.DataFrame(rows)

    # compute macro averages
    macro_precision = np.mean(df['Precision'])
    macro_recall = np.mean(df['Recall'])
    macro_f1 = np.mean(df['F1'])
    
    # compute micro averages
    micro_precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    micro_recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    
    # compute weighted averages
    weighted_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    weighted_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    averages = pd.DataFrame([ 
        {'Disease': 'MACRO AVG', 'Precision': macro_precision, 'Recall': macro_recall, 'F1': macro_f1, 'Support': ''},
        {'Disease': 'MICRO AVG', 'Precision': micro_precision, 'Recall': micro_recall, 'F1': micro_f1, 'Support': ''},
        {'Disease': 'WEIGHTED AVG', 'Precision': weighted_precision, 'Recall': weighted_recall, 'F1': weighted_f1, 'Support': ''},
    ])

    display_df = pd.concat([df, averages], ignore_index=True)
    for col in ['Precision', 'Recall', 'F1']:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.floating)) else x)

    print("\n" + "="*70)
    print("PER-CLASS METRICS")
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


def display_results(metrics, checkpoint_name, all_predictions, all_labels, all_probabilities, test_dataset):
    
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"EVALUATION RESULTS - {checkpoint_name}")
    print(f"\nOverall Metrics:")
    print(f"  Error: {metrics['error']:.4f} ({metrics['error']*100:.2f}%)")
    print(f"  Loss: {metrics['loss']:.4f}")
    
    # np.random.seed(RANDOM_SEED)
    idx = np.random.choice(len(test_dataset), 1)[0]
    
    # get data for random sample
    img, _ = test_dataset[idx]
    prediction = all_predictions[idx]
    truth = all_labels[idx]
    probability = all_probabilities[idx]
    is_correct = np.array_equal(prediction, truth)
    
    # convert tensor to displayable format
    img = img.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
        
    # build text with predictions and ground truth to display on image
    lines = []
    lines.append(f"Sample {idx} | {'CORRECT' if is_correct else 'INCORRECT'}")
    lines.append("")
    lines.append("Disease   | GT | Pred | Prob")
    lines.append("-" * 35)

    for j, name in enumerate(CLASS_NAMES):
        match = "OK" if prediction[j] == truth[j] else "XX"
        lines.append(f"{CLASS_SHORT[j]:<9} | {truth[j]:<2} | {prediction[j]:<4} | {probability[j]:.3f} {match}")

    lines.append("")
    lines.append("GT:  " + ", ".join([CLASS_SHORT[j] for j, v in enumerate(truth) if v == 1]) or "None")
    lines.append("Pred: " + ", ".join([f"{CLASS_SHORT[j]}({probability[j]:.2f})" for j, v in enumerate(prediction) if v == 1]) or "None")

    text_to_display = "\n".join(lines)

    # create image with fundus sample
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))

    # Display image
    ax.imshow(img)
    ax.axis('off')

    # Add text box with predictions
    ax.text(
        0.5, -0.15,
        text_to_display,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='center',
        family='monospace'
    )

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sample_prediction.png'), bbox_inches='tight')
    plt.show()

    print(f"\nImage saved to: {RESULTS_DIR}/sample_prediction.png")

def compute_and_plot_confusion_matrix(all_labels, all_preds, class_names, save_dir=RESULTS_DIR):

    os.makedirs(save_dir, exist_ok=True)
    
    n_classes = len(class_names)
    metrics_summary = {}
    
    # plot confusion matrices for each disease class
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, name in enumerate(class_names):
        # compute confusion matrix
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        tn, fp, fn, tp = cm.ravel()
        
        # calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # store metrics
        metrics_summary[name] = {
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                   cbar=False)
        axes[i].set_title(f'{name}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel(f'Predicted\nP:{precision:.3f} R:{recall:.3f} F1:{f1:.3f}', fontsize=8)
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_individual.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # plot overall confusion matrix
    all_preds_flat = all_preds.ravel()
    all_labels_flat = all_labels.ravel()
    cm_overall = confusion_matrix(all_labels_flat, all_preds_flat)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    ax.set_title('Overall Confusion Matrix (All Classes Combined)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_overall.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # print summary table of confusion matrix metrics
    print("\n" + "="*70)
    print("CONFUSION MATRIX SUMMARY")
    print("="*70)
    print(f"{'Disease':<25} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print("-"*95)
    
    for name, m in metrics_summary.items():
        print(f"{name:<25} {m['TP']:<6} {m['FP']:<6} {m['FN']:<6} {m['TN']:<6} "
              f"{m['precision']:.4f}  {m['recall']:.4f}  {m['f1']:.4f}")
    
    print("-"*95)
    
    return metrics_summary

def run_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using: {device}")
    
    # load test data
    test_dataset = OcularDataset(TEST_CSV, IMAGE_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    
    # load model
    model = load_checkpoint(CHECKPOINT_FILE, MODEL_CLASS, MODULE_PATH)
    model = model.to(device)
    
    # evaluate
    metrics, all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device, THRESHOLD)
    
    # display confusion matrix metrics
    compute_and_plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES, RESULTS_DIR)

    print_per_class_metrics(all_labels, all_preds, CLASS_NAMES)

    # display a sample result
    display_results(metrics, CHECKPOINT_FILE, all_preds, all_labels, all_probs, test_dataset)

if __name__ == "__main__":
    run_evaluate()