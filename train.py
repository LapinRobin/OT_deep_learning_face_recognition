import torch
import torch.nn as nn
import torch.optim as optim
from load_data import train_loader, valid_loader
from net import Net, Net2
import time
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

if __name__ == '__main__':

    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')

    # Hyperparameters
    n_epochs = 5
    learning_rate = 0.0005

    # Initialize the model
    model = Net2().to(device)
    
    # Add class weights to penalize false negatives more heavily
    # [noface_weight, face_weight] 
    class_weights = torch.FloatTensor([0.1, 12.0]).to(device)  # Increased weight for face class to penalize false negatives more
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=2,
                                                    verbose=True)

    # Initialize AdaBoost
    base_classifier = DecisionTreeClassifier(max_depth=3)
    ada_boost = AdaBoostClassifier(
        estimator=base_classifier,
        n_estimators=50,
        learning_rate=1.0,
        algorithm='SAMME'
    )

    # Modify the training loop to collect features for AdaBoost
    all_features = []
    all_labels = []

    # Training loop
    best_valid_loss = float('inf')

    # Create progress bar for epochs
    epoch_pbar = tqdm(range(n_epochs), desc='Training Progress', position=0)

    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        start_time = time.time()
        
        # Create progress bar for training batches
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f'Epoch {epoch+1}/{n_epochs} [Train]', 
                         position=1, leave=False)
        
        for batch_idx, (data, target) in train_pbar:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            
            # Store the output probabilities for AdaBoost
            probs = torch.softmax(output, dim=1).detach().cpu().numpy()
            all_features.append(probs)
            all_labels.append(target.cpu().numpy())
            
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            train_loss += loss.item()
            
            # Update progress bar with current metrics
            current_accuracy = 100 * train_correct / train_total
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{current_accuracy:.2f}%'
            })
        
        # Validation phase
        model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_total = 0
        
        # Create progress bar for validation batches
        valid_pbar = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Valid]', 
                         position=1, leave=False)
        
        with torch.no_grad():
            for data, target in valid_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                valid_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                valid_total += target.size(0)
                valid_correct += (predicted == target).sum().item()
                
                # Update validation progress bar
                current_valid_accuracy = 100 * valid_correct / valid_total
                valid_pbar.set_postfix({
                    'loss': f'{valid_loss/valid_total:.4f}',
                    'accuracy': f'{current_valid_accuracy:.2f}%'
                })
        
        # After validation phase, add scheduler step
        scheduler.step(valid_loss)
        
        # Calculate average losses and accuracies
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        train_accuracy = 100 * train_correct / train_total
        valid_accuracy = 100 * valid_correct / valid_total
        epoch_time = time.time() - start_time
        
        # Update epoch progress bar with summary
        epoch_pbar.set_postfix({
            'time': f'{epoch_time:.2f}s',
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_accuracy:.2f}%',
            'valid_loss': f'{valid_loss:.4f}',
            'valid_acc': f'{valid_accuracy:.2f}%'
        })
        
        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
            tqdm.write('New best model saved')

        # Train AdaBoost after collecting all features
        if epoch == n_epochs - 1:  # Train AdaBoost on the final epoch
            X = np.vstack(all_features)
            y = np.concatenate(all_labels)
            ada_boost.fit(X, y)
            print("AdaBoost training completed!")

    print('\nTraining completed!')