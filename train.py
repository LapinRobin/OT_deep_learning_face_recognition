import torch
import torch.nn as nn
import torch.optim as optim
from load_data import train_loader, valid_loader
from net import Net
import time

if __name__ == '__main__':
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')

    # Print dataset information
    print("\nDataset Information:")
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"Input batch shape: {sample_batch.shape}")
    print(f"Labels shape: {sample_labels.shape}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(valid_loader)}")

    # Hyperparameters
    n_epochs = 10
    learning_rate = 0.001
    print(f"\nHyperparameters:")
    print(f"Number of epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")

    # Initialize the model
    print("\nInitializing model...")
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Model initialized successfully")

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training loop
    best_valid_loss = float('inf')

    for epoch in range(n_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"{'='*50}")
        
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"\nProcessing batch {batch_idx+1}/{len(train_loader)}")
            print(f"Batch input shape: {data.shape}")
            print(f"Batch target shape: {target.shape}")
            
            data, target = data.to(device), target.to(device)
            print(f"Data moved to device: {device}")
            
            # Forward pass
            optimizer.zero_grad()
            try:
                output = model(data)
                print(f"Forward pass successful. Output shape: {output.shape}")
            except Exception as e:
                print(f"Error in forward pass: {str(e)}")
                raise
            
            loss = criterion(output, target)
            print(f"Loss calculated: {loss.item():.4f}")
            
            # Backward pass
            loss.backward()
            optimizer.step()
            print("Backward pass completed")
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                current_accuracy = 100 * train_correct / train_total
                print(f'Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Running Accuracy: {current_accuracy:.2f}%')
        
        # Validation phase
        print("\nStarting validation phase...")
        model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                print(f"Validating batch {batch_idx+1}/{len(valid_loader)}")
                data, target = data.to(device), target.to(device)
                try:
                    output = model(data)
                    valid_loss += criterion(output, target).item()
                    _, predicted = torch.max(output.data, 1)
                    valid_total += target.size(0)
                    valid_correct += (predicted == target).sum().item()
                except Exception as e:
                    print(f"Error during validation: {str(e)}")
                    raise
        
        # Calculate average losses and accuracies
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        train_accuracy = 100 * train_correct / train_total
        valid_accuracy = 100 * valid_correct / valid_total
        epoch_time = time.time() - start_time
        
        print(f'\nEpoch {epoch+1}/{n_epochs} Summary:')
        print(f'Time taken: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')
        
        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('New best model saved as best_model.pth')

    print('\nTraining completed!') 