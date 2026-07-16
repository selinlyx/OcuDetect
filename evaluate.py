import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import importlib
from train import OcularDataset

MODEL_CLASS = "BaselineModel"
MODULE_PATH = "models.baseline" 

CHECKPOINT_FILE = "Baseline_epochs19_bs32_lr0.001.pt"

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


def display_results(metrics, checkpoint_name, all_predictions, all_labels, all_probabilities, test_dataset):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"EVALUATION RESULTS - {checkpoint_name}")
    print(f"\nMetrics:")
    print(f"  Error: {metrics['error']:.4f} ({metrics['error']*100:.2f}%)")
    print(f"  Loss: {metrics['loss']:.4f}")
    
    print("SAMPLE PREDICTIONS")
    np.random.seed(RANDOM_SEED)
    
    # pick up to 5 indices from test dataset to evaluate on 
    indices = np.random.choice(len(test_dataset), min(5, len(test_dataset)), replace=False)
        
    for i, idx in enumerate(indices):
        prediction = all_predictions[idx]
        truth = all_labels[idx]
        probability = all_probabilities[idx]
        is_correct = np.array_equal(prediction, truth) # check if all diseases match
        
        print(f"\nSample {i+1} (index {idx}): {'FULL MATCH' if is_correct else 'INCORRECT/INCOMPLETE PREDICTION'}")
        
        for j, disease in enumerate(CLASS_NAMES): # check for individual disease matches
            match = "✅" if prediction[j] == truth[j] else "❌"
            print(f"{disease} ({CLASS_SHORT[j]}): Ground Truth = {truth[j]}, Prediction = {prediction[j]}, Probability = {probability[j]:.3f} {match}")
                
    
    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 4 * len(indices)))
    
    # if only one sample, axes is 1D, so make it 2D
    if len(indices) == 1:
        axes = axes.reshape(1, 2)
    
    for i, idx in enumerate(indices):
        # get the image from dataset
        img, _ = test_dataset[idx]
        
        # convert tensor to displayable format
        img = img.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # get predictions for this sample
        prediction = all_predictions[idx]
        truth = all_labels[idx]
        is_correct = np.array_equal(prediction, truth)
        
        gt_text = "Ground Truth:\n"
        for j, disease in enumerate(CLASS_NAMES):
            gt_text += f"{CLASS_SHORT[j]}: {truth[j]} "
        
        prediction_text = "Prediction:\n"
        for j, disease in enumerate(CLASS_NAMES):
            prediction_text += f"{CLASS_SHORT[j]}: {prediction[j]} "
        
        # left column: ground truth image
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Sample {idx}\nGround Truth\n{gt_text}", fontsize=8)
        
        # right column: prediction image
        axes[i, 1].imshow(img)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"Sample {idx}\nPrediction {'(CORRECT)' if is_correct else '(INCORRECT)'}\n{prediction_text}", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sample_predictions.png'))
    plt.show()
    
    print(f"\nImages saved to: {RESULTS_DIR}/sample_predictions.png")

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
    
    # display results
    display_results(metrics, CHECKPOINT_FILE, all_preds, all_labels, all_probs, test_dataset)

if __name__ == "__main__":
    run_evaluate()