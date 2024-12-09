import torch
import numpy as np
from net import Net
from load_data import test_loader, classes
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix using seaborn."""
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def test_model(model, device, test_loader):
    model.eval()
    
    # Lists to store predictions and true labels
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    
    print("\nStarting model evaluation...")
    
    # Use tqdm for progress bar
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Get predicted class
            probabilities = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            
            # Store predictions and true labels
            all_predictions.extend(pred.cpu().numpy())
            all_true_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions).flatten()
    all_true_labels = np.array(all_true_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average='weighted')
    recall = recall_score(all_true_labels, all_predictions, average='weighted')
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    # Calculate ROC AUC score
    try:
        roc_auc = roc_auc_score(all_true_labels, all_probabilities[:, 1])
    except:
        roc_auc = None
    
    # Get detailed classification report
    class_report = classification_report(all_true_labels, all_predictions, target_names=classes)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nOverall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC Score: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nDetailed Classification Report:")
    print(class_report)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, classes)
    print("\nConfusion matrix has been saved as 'confusion_matrix.png'")
    
    # Calculate per-class accuracy
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(classes):
        class_correct = cm[i, i]
        class_total = cm[i].sum()
        class_accuracy = class_correct / class_total
        print(f"{class_name}: {class_accuracy:.4f} ({class_correct}/{class_total})")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': class_report
    }

if __name__ == '__main__':
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f'Using device: {device}')
    
    # Load the model
    model = Net().to(device)
    try:
        model.load_state_dict(torch.load('best_model.pth'))
        print("Successfully loaded model from 'best_model.pth'")
    except:
        print("Error: Could not load model. Make sure 'best_model.pth' exists.")
        exit(1)
    
    # Test the model
    metrics = test_model(model, device, test_loader)

