#!/usr/bin/python3

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoTokenizer
import torch.optim as optim

# Define Model A (produces floating-point tensors)
class ModelA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ModelA, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

# Define the STE (Straight-Through Estimator) for discretization
class DiscretizationWithSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)  # Discretize (round to nearest integer)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Straight-through gradient during backpropagation

discretize = DiscretizationWithSTE.apply

# Create a toy dataset (input features)
def generate_toy_data(num_samples, input_dim):
    return torch.randn(num_samples, input_dim)

# Main training script
def train_model():
    # Toy data settings
    input_dim = 10  # Dimensions of input data
    output_dim = 30522  # Output dimensions (this will map to BERT token IDs; BERT's vocabulary size is 30,522)
    hidden_dim = 32  # Hidden layer size for Model A
    num_samples = 32  # Number of samples in the toy dataset
    epochs = 50  # Number of training epochs

    # Generate toy input data
    toy_data = generate_toy_data(num_samples, input_dim)

    # Load frozen BERT model
    bert_model = BertModel.from_pretrained("./HLT/models/google-bert/bert-large-uncased")
    bert_model.eval()  # Freeze BERT weights
    
    # Tokenizer for BERT
    tokenizer = AutoTokenizer.from_pretrained("./HLT/models/google-bert/bert-large-uncased")

    # Instantiate Model A
    model_a = ModelA(input_dim, hidden_dim, output_dim)
    
    # Define an optimizer for Model A
    optimizer = optim.Adam(model_a.parameters(), lr=1e-3)
    
    # Define a simple loss function
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model_a.train()
        
        # Forward pass through Model A
        optimizer.zero_grad()
        float_output = model_a(toy_data)  # Model A produces continuous values
        
        # Apply STE to discretize the output (round to nearest integer)
        discretized_output = discretize(float_output)

        # Convert discretized token IDs to BERT input format (add batch dimension)
        input_ids = discretized_output.long().clamp(0, tokenizer.vocab_size - 1)  # Ensure valid token IDs
        input_ids = input_ids.unsqueeze(0)  # Add batch dimension
        
        # Feed into frozen BERT
        with torch.no_grad():
            bert_outputs = bert_model(input_ids=input_ids)
            bert_embeddings = bert_outputs.last_hidden_state  # Use the last hidden state

        # Calculate a toy loss (here, minimizing the mean of the BERT embeddings)
        # You can choose a more meaningful loss depending on the task
        loss = loss_fn(bert_embeddings.mean(dim=1), torch.zeros_like(bert_embeddings.mean(dim=1)))

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Print loss for monitoring
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

if __name__ == "__main__":
    train_model()

