import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# Import your model, dataset, and config
from V2.models.ResnetSesimic import ResNetSeismic, model  # Ensure the model is loaded
from V2.models.SeismicDataset import train_loader  # DataLoader

print(next(model.parameters()).device)


# Verify that the device is set correctly (cuda for GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Model is on: {next(model.parameters()).device}")

# Ensure the model is on the correct device
model = ResNetSeismic().to(device)

# Define the loss function and optimizer
criterion_start = nn.MSELoss()  # MSE loss for start time prediction
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):

    running_loss_start = 0.0

    for i, (spectrograms, start_labels) in tqdm(enumerate(train_loader)):
        model.train()
        # Move data to GPU
        spectrograms = spectrograms.to(device)  # Ensure input is on GPU
        start_labels = start_labels.to(device).unsqueeze(1)  # Ensure labels are on GPU

        # Forward pass
        start_output = model(spectrograms)

        # Compute loss
        loss_start = criterion_start(start_output, start_labels)


        # Track the running loss
        running_loss_start += loss_start.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss_start.backward()
        optimizer.step()


    print(f"Epoch [{epoch+1}/{num_epochs}], Start Loss: {running_loss_start/i:.4f}")

torch.save(obj=model.state_dict(),
           f='models/model.pth')
