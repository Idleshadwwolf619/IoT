import os
import tkinter as tk
from tkinter import scrolledtext
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Import llama_cpp for loading the model
from llama_cpp import Llama

# Define the model architecture
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Define a custom dataset class for loading and preprocessing data
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def generate_iot_data(num_samples, num_features):
    # Generate synthetic sensor data
    iot_data = np.random.rand(num_samples, num_features)
    return iot_data

def inference_on_iot_data(iot_data, neural_model):
    # Convert data to PyTorch tensor
    iot_data_tensor = torch.tensor(iot_data, dtype=torch.float32)
    
    # Perform inference
    neural_model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = neural_model(iot_data_tensor)
        _, predicted = outputs.max(1)
        predictions = predicted.tolist()  # Convert predictions to list
    return predictions

# Define training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, running_loss))

def load_model(model_path):
    if not os.path.isfile(model_path):
        print("Error: Model file not found at the specified path:", model_path)
        return None
    model = Llama(model_path=model_path, seed=42)  # Load the model using llama_cpp
    return model.metadata  # Return model metadata attribute

def load_and_preprocess_data(model_metadata):
    # Extract input size from model metadata
    input_size = int(model_metadata['llama.embedding_length'])

    # Load dummy data
    train_data = np.random.rand(100, input_size)  # Assuming input size is 4096
    train_labels = np.random.randint(0, 2, size=(100,))  # Assuming binary classification
    val_data = np.random.rand(20, input_size)
    val_labels = np.random.randint(0, 2, size=(20,))
    
    # Convert data to PyTorch tensors
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_data_tensor = torch.tensor(val_data, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    
    return train_data_tensor, train_labels_tensor, val_data_tensor, val_labels_tensor

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        # Train the model
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluate the model on validation data
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total

        # Print validation metrics
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    print('Finished Training')

def send_message(text_area_main_user_input, text_area_display, llm_model):
    # Get user input from the text area
    user_input = text_area_main_user_input.get("1.0", tk.END).strip()  # Extract user input

    if user_input:  # Check if the user input is not empty
        # Generate response using the LLM model
        llm_response = generate_llm_response(user_input, llm_model)

        # Display the LLM model's response in the text area
        text_area_display.insert(tk.END, "\n\nUser: " + user_input)  # Display user input
        text_area_display.insert(tk.END, "\nLLM Model: " + llm_response)  # Display LLM model response
        text_area_main_user_input.delete("1.0", tk.END)  # Clear user input area

# Define a function to generate response using the LLM model
def generate_llm_response(prompt, llm_model):
    # Tokenize the prompt
    tokenized_prompt = llm_model.tokenize(prompt.encode("utf-8"))

    # Generate response from the model
    response_tokens = []
    for token in llm_model.generate(tokenized_prompt, top_k=40, top_p=0.95, temp=0.72, repeat_penalty=1.1):
        response_tokens.append(token)
        if token == llm_model.token_eos():
            break

    # Decode the response tokens
    llm_response = llm_model.detokenize(response_tokens)

    # Decode the response from bytes to string
    llm_response = llm_response.decode("utf-8")

    return llm_response

def main(model_metadata):
    
    llm_model = Llama(model_path="dolphin-2.6-mistral-7b-dpo.Q5_K_M.gguf", seed=42)

    # Define model parameters
    batch_size = 32
    learning_rate = 0.001

    # Load the LLM model and obtain model metadata
    llm_model_metadata = model_metadata
    if llm_model_metadata is None:
        return  # Exit if model loading fails

    # Load and preprocess data using model metadata
    train_data, train_labels, val_data, val_labels = load_and_preprocess_data(llm_model_metadata)
    train_dataset = CustomDataset(train_data, train_labels)
    val_dataset = CustomDataset(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)  # Define validation data loader
    
    # Define neural network model, loss function, and optimizer
    input_size = int(llm_model_metadata['llama.embedding_length'])  # Extract input size from model metadata
    output_size = 2  # Replace '2' with your actual output size
    neural_model = SimpleModel(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(neural_model.parameters(), lr=learning_rate)

    # Create the main window of the GUI
    root = tk.Tk()
    root.title("Deep Learning GUI")

    # Create GUI components
    text_area_display = scrolledtext.ScrolledText(root, width=80, height=20)
    text_area_display.pack()

    # Create a frame for the user to input text as a prompt for the model
    frame_main_user_input = tk.Frame(root)
    scrollbar_main_user_input = tk.Scrollbar(frame_main_user_input)

    global text_area_main_user_input
    text_area_main_user_input = scrolledtext.ScrolledText(frame_main_user_input, width=128, height=5, yscrollcommand=scrollbar_main_user_input.set)
    # Set the background color and the foreground color of the text area, and the font.
    # Change the colors to match your application's color scheme
    text_area_main_user_input.config(background="#202020", foreground="#ffff33", font=("Courier", 12))
    scrollbar_main_user_input.config(command=text_area_main_user_input.yview)
    # Fill out the root window with the frame
    text_area_main_user_input.pack(side=tk.LEFT, fill=tk.BOTH)
    scrollbar_main_user_input.pack(side=tk.RIGHT, fill=tk.Y)
    frame_main_user_input.pack()

    def train_neural_model():
        # Train and evaluate the neural network model
        train_and_evaluate(neural_model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
    
    def send_message_and_get_llm_response():
        send_message(text_area_main_user_input, text_area_display, llm_model)

    def generate_and_display_iot_data():
        # Generate IoT data
        iot_data = generate_iot_data(10, input_size)  # Assuming 10 samples
        # Perform inference on IoT data
        predictions = inference_on_iot_data(iot_data, neural_model)
        # Display the predictions
        text_area_display.insert(tk.END, "\n\nIoT Data Predictions:")
        text_area_display.insert(tk.END, "\n" + str(predictions))

    train_button = tk.Button(root, text="Train Model", command=train_neural_model)
    train_button.pack()

    send_button = tk.Button(root, text="Send", command=send_message_and_get_llm_response)
    send_button.pack()

    iot_button = tk.Button(root, text="Generate IoT Data", command=generate_and_display_iot_data)
    iot_button.pack()

    root.mainloop()

# Call the main function and pass model_metadata
if __name__ == "__main__":
    # Define model parameters
    model_path = "dolphin-2.6-mistral-7b-dpo.Q5_K_M.gguf"

    # Load the model and obtain model metadata
    model_metadata = load_model(model_path)
    if model_metadata is None:
        exit(1)  # Exit if model loading fails

    main(model_metadata)
