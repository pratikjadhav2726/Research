import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from Lrp_utils import LRPDropout

def newlayer(layer, g):
    """Clone a layer and pass its parameters through the function g."""
    layer = copy.deepcopy(layer)
    if isinstance(layer, torch.nn.MaxPool2d):
      return layer
    else:
      layer.weight = torch.nn.Parameter(g(layer.weight))
      layer.bias = torch.nn.Parameter(g(layer.bias))
      return layer



# Step 4: Train the Model
def train_model(model,num_epochs,train_loader,train_on_gpu,optimizer,criterion,valid_loader):
    train_losses = []  # List to store training losses
    valid_losses = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for images, labels in train_loader:
            if train_on_gpu:
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # Validation
        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:
                if train_on_gpu:
                    images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * images.size(0)

        # Calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        # Append the losses to the lists
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # Print training and validation losses
        print(f'Epoch {epoch+1}/{num_epochs} \t Training Loss: {train_loss:.4f} \t Validation Loss: {valid_loss:.4f}')
    # Plot the training and validation losses
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), valid_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)  # Set y-axis limits
    plt.title("Training and Validation Losses Random 20%")
    plt.legend()
    plt.show()

# Step 5: Validate the Model (Included in Training Function)

# Step 6: Test the Model
def test_model(model,test_loader,criterion, train_on_gpu):
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            if train_on_gpu:
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader.dataset)
    accuracy = (correct / total) * 100
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')




def LRP_individual(model, X, target, device):
  model.eval()
  # Get the list of layers of the network
  layers = [module for module in model.modules() if not isinstance(module, torch.nn.Sequential)][1:]
  # print(layers)
  # Propagate the input
  L = len(layers)
  A = [X] + [X] * L # Create a list to store the activation produced by each layer
  # print(layers)
  # print(len(A),layers)
  for layer in range(L):
      # print(A[layer].shape,layers[layer])
      # print(layers[layer])
      # if isinstance(layers[layer],torch.nn.Linear):
      #   # print(layers[layer])
      #   A[layer]=A[layer].reshape(-1,1,layers[layer].in_features)
      # print(A[layer].shape,layers[layer])
      if isinstance(layers[layer], LRPDropout):
        # print(A[layer].shape)
        A[layer + 1] =A[layer]
      else:
        A[layer + 1] = layers[layer].forward(A[layer])

  # Get the relevance of the last layer using the highest classification score of the top layer
  T = A[-1].to(device)  # Remove .numpy().tolist()
  index = target
  # print(T.shape)
  T = torch.abs(T) * 0
  # print(T)
  T[-1, index] = 1  # Modify to index the element directly
  T = T.to(device)
  # Create the list of relevances with (L + 1) elements and assign the value of the last one
  R = [None] * L + [(A[-1] * T).data + 1e-6]
  # print("hi")
    # Propagation procedure from the top-layer towards the lower layers
  for layer in range(0, L)[::-1]:

      if isinstance(layers[layer], torch.nn.Conv2d) or isinstance(layers[layer], torch.nn.Conv3d) \
              or isinstance(layers[layer],torch.nn.Linear) or isinstance(layers[layer],torch.nn.MaxPool2d) :


          rho = lambda p: p

          A[layer] = A[layer].data.requires_grad_(True).to(device)

          # Step 1: Transform the weights of the layer and executes a forward pass
          z = newlayer(layers[layer], rho)

          z=z.forward(A[layer]) + 1e-9
          # print(layers[layer],z.shape,A[layer].shape)
          # print(z.shape,R[layer+1].shape,layers[layer],A[layer].shape)
          # Step 2: Element-wise division between the relevance of the next layer and the denominator
          # print(R[layer+1].shape, z.shape,layer+1)
          s = (R[layer+1] / z).data
          # print(s)
          # Step 3: Calculate the gradient and multiply it by the activation layer
          (z * s).sum().backward()
          c = A[layer].grad
          R[layer] = (A[layer] * c).cuda().data
          # R[layer] = R[layer + 1]
          # print(R)

      else:
          # print(layers[layer],"else")
          R[layer] = R[layer + 1]
      # if layer == 10:
      #             # print("hi")
      #             R[layer] = R[layer].reshape(-1,512,4,4)
  # Return the relevance of the all the layers
  model.train()
  return R



  

  # Step 4: Train the Model
def train_modellrp(model,num_epochs,train_loader,valid_loader,train_on_gpu,optimizer,criterion):
    train_losses = []  # List to store training losses
    valid_losses = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for images, labels in train_loader:
            # print(images.size())
            if train_on_gpu:
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            Rel = LRP_individual(model, images.reshape(-1,28*28).float().to("cuda"),labels,device="cuda")
            # [print(tensor.shape,end="...") for tensor in Rel]
            # print()
            avg_tensor=torch.tensor([])
            for i in range(len(Rel)):
                if(len(avg_tensor)==0):
                    avg_tensor=Rel
                    for j in range(len(Rel)):
                        avg_tensor[j] = torch.mean(Rel[j], dim=0)
            avg_tensor[i] = torch.mean(Rel[i], dim=0)
            # [print(tensor.shape,end="...") for tensor in avg_tensor]
            model.dropout1.update_mask(avg_tensor[1])
            model.dropout2.update_mask(avg_tensor[3])
            # # print("Hi")
            model.dropout3.update_mask(avg_tensor[5])
            # Validation
        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:
                if train_on_gpu:
                    images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * images.size(0)

        # Calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        # Append the losses to the lists
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # Print training and validation losses
        print(f'Epoch {epoch+1}/{num_epochs} \t Training Loss: {train_loss:.4f} \t Validation Loss: {valid_loss:.4f}')
    # Plot the training and validation losses
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), valid_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)  # Set y-axis limits
    plt.title("Training and Validation Losses Highdrop 50%")
    plt.legend()
    plt.show()
