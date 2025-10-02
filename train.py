import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torchsummary import summary

from model_1 import CNN_Model as CNN_Model_1, get_optimizer as get_optimizer_1, get_optimizer_params as get_optimizer_params_1, get_train_transform as get_train_transform_1, get_scheduler as get_scheduler_1, get_scheduler_params as get_scheduler_params_1
from model_2 import CNN_Model as CNN_Model_2, get_optimizer as get_optimizer_2, get_optimizer_params as get_optimizer_params_2, get_train_transform as get_train_transform_2, get_scheduler as get_scheduler_2, get_scheduler_params as get_scheduler_params_2
from model_3 import CNN_Model as CNN_Model_3, get_optimizer as get_optimizer_3, get_optimizer_params as get_optimizer_params_3, get_train_transform as get_train_transform_3, get_scheduler as get_scheduler_3, get_scheduler_params as get_scheduler_params_3

from utils import compute_mean_std, data_loader, initialize_model, save_plot_metrics, save_train_test_metrics, train_model

print("Starting training script...")

# Model selection
print("\n" + "="*50)
print("Model Selection")
print("="*50)
print("Available models:")
print("1. CNN_Model_1 - 2 million parameters, 60 epochs to reach >85% accuracy")
print("2. CNN_Model_2 - 250k parameters, 70 epochs to reach >85% accuracy")
print("3. CNN_Model_3 - 100k parameters, 75 epochs to reach >85% accuracy")

while True:
    try:
        choice = input("\nSelect model to train (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please select a valid option.")
    except KeyboardInterrupt:
        print("\nTraining cancelled by user.")
        exit(0)

# Map choice to model class, name, and optimizer functions
model_mapping = {
    '1': (CNN_Model_1, 'CNN_Model_1', get_optimizer_1, get_optimizer_params_1, get_train_transform_1, get_scheduler_1, get_scheduler_params_1),
    '2': (CNN_Model_2, 'CNN_Model_2', get_optimizer_2, get_optimizer_params_2, get_train_transform_2, get_scheduler_2, get_scheduler_params_2),
    '3': (CNN_Model_3, 'CNN_Model_3', get_optimizer_3, get_optimizer_params_3, get_train_transform_3, get_scheduler_3, get_scheduler_params_3),
}

selected_model_class, selected_model_name, get_optimizer_func, get_optimizer_params_func, get_train_transform_func, get_scheduler_func, get_scheduler_params_func = model_mapping[choice]
print(f"\nSelected: {selected_model_name}")
print("="*50)

# Check if CUDA is available
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# Compute mean and std of training data
print("Computing mean and std of training data...")
train_mean, train_std = compute_mean_std()
print("Computation done.")
print(f"Mean: {train_mean}, Std: {train_std}")

# Data loaders
print("Preparing data loaders...")
train_transform = get_train_transform_func(train_mean, train_std)
train_loader, test_loader = data_loader(train_mean=train_mean, train_std=train_std, batch_size_train=2048, batch_size_test=2048, train_transform=train_transform)
print("Data loaders ready.")

# Initialize model, loss function, optimizer
print("Initializing model...")
optimizer_func = get_optimizer_func()
optimizer_params = get_optimizer_params_func()
model, criterion, optimizer, device = initialize_model(selected_model_class, optimizer_func, **optimizer_params)
print("Model initialized successfully.")

# Print model summary
print("Model Summary:")
summary(model, input_size=(3, 32, 32)) # Since input size for CIFAR-10 is 32x32 with 3 channels

# Model Training
epochs = int(input("\nSelect number of epochs for training (default 20): ") or 20)
print(f"Starting training for {selected_model_name}...")

# Initialize scheduler for model training
scheduler = None
scheduler_func = get_scheduler_func()
if choice in ["2", "3"]:
    scheduler_params = get_scheduler_params_func(len(train_loader), epochs=epochs)
else:
    scheduler_params = get_scheduler_params_func()
scheduler = scheduler_func(optimizer, **scheduler_params)
print("Scheduler initialized for model training.")

# Actual model training
train_losses, train_accuracies, test_losses, test_accuracies = train_model(model, device, train_loader, test_loader, optimizer, 
                                                                           criterion, epochs=epochs, scheduler=scheduler)
print("Training completed!")

# Save the train and test losses and accuracies for future reference
print("Saving training and testing metrics...")
save_train_test_metrics(train_losses, train_accuracies, test_losses, test_accuracies, selected_model_name)
print("Metrics saved successfully.")

# Save plot metrics for visualization
print("Saving plot metrics...")
save_plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies, selected_model_name)
print("Plot metrics saved successfully.")

# Print final statistics
print(f"\nFinal Results:")
print(f"Training Loss: {train_losses[-1]:.4f}")
print(f"Training Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Test Loss: {test_losses[-1]:.4f}")
print(f"Test Accuracy: {test_accuracies[-1]:.2f}%")
