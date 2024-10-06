import argparse
import torch
from torchvision import models
from torch import nn, optim
import torchvision
from tqdm import tqdm
from torchvision import datasets, transforms
from collections import OrderedDict
import os


    
def load_data(data_dir,arch):
    print(f"Loading data from directory: {data_dir}")
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define your data transformations
    train_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomRotation(degrees=(-15, 15), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if arch== 'vit16':
        from torchvision.models import ViT_B_16_Weights
        weights = ViT_B_16_Weights.DEFAULT
        default_transforms = weights.transforms()
    elif arch == 'vgg13':
        from torchvision.models import VGG13_Weights
        weights = VGG13_Weights.DEFAULT
        default_transforms = weights.transforms()
    else:
        print("Architecture is",arch)
        raise ValueError('Invalid architecture. Choose vit16 or vgg13.')
    
    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, default_transforms)
    test_dataset = datasets.ImageFolder(test_dir, default_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, valid_loader,test_loader,train_dataset

# model, criterion, optimizer = 
def create_model(arch, hidden, learn_rate, gpu):
    if arch == 'vgg13':
        print("Using VGG13 model")
        vgg13_weights = torchvision.models.VGG13_Weights.DEFAULT
        model = models.vgg13(weights=vgg13_weights)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(
        nn.Linear(25088, hidden),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden, 102),
        nn.LogSoftmax(dim=1)
    )
    elif arch == 'vit16':
        print("Using Vision Transformer (ViT16) model")
        HIDDEN_LAYER_1 = 256
        HIDDEN_LAYER_2 = 128
        vit_b_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        vit_b_default_transforms = vit_b_weights.transforms()
        model = torchvision.models.vit_b_16(weights=vit_b_weights)
        for param in model.parameters():
            param.requires_grad = False
        model.heads = nn.Sequential(
            nn.Linear(768, HIDDEN_LAYER_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_1, HIDDEN_LAYER_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_2,102)
        )
    else:
        print("No such model, select either 'vgg16' or 'vgg13'.")
        exit()
        
    return model
    
    
def test_model(model,test_loader,criterion,device):
    model.eval()
    test_loss=0.0
    correct,total=0,0
    with torch.no_grad():
        for inputs,labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            pred = model(inputs)
            loss = criterion(pred, labels)
            test_loss += loss.item()
            _, predicted = pred.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return test_loss / len(test_loader), 100 * correct / total
   
def save_model(model: nn.Module, model_name: str) -> str:
    '''
    Function for saving the state_dict of the model.

    Args:
        model: The model (nn.Module) whose state_dict you want to save.
        model_name: The name to use for saving the state_dict.

    Returns:
        The path to the saved state_dict (str).
    '''

    # Model save location
    SAVE_PATH = f"./{model_name}.pt"

    # Saving model state_dict and printing message
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"[INFO] {model_name} saved to {SAVE_PATH}")

    # Returning path of saved model
    return SAVE_PATH
  
def save_checkpoint(model, save_dir, arch, hidden, epochs, learn_rate, train_data):
    model.class_to_idx = train_data.class_to_idx
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    checkpoint = {
        'architecture': arch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'hidden_layers': hidden,
        'epochs': epochs,
        'learning_rate': learn_rate
    }
    
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint-{arch}.pth")
    torch.save(checkpoint, checkpoint_path)

    # Printing save confirmation
    print(f"[INFO] The general checkpoint has been saved to: {checkpoint_path}")
    return checkpoint_path


#train_loader, valid_loader,test_loader,train_dataset, args.epochs, args.learning_rate, args.arch,args.save_dir
def train_model(train_loader, valid_loader,test_loader,train_data, epochs, learning_rate,hidden,arch, save_dir,gpu=False):
    print("Starting training...")
    print(f"Model: {arch},  Learn Rate: {learning_rate}, Epochs: {epochs}")
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    # clearing cache if using GPU
    if str(device) == "cuda":
        torch.cuda.empty_cache()
    
    #create model
    model=create_model(arch, hidden,learning_rate, gpu)
    if gpu:
        model = model.to(device)
        
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc="Training")
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=running_loss / len(train_loader))

        train_loss=running_loss/len(train_loader)
        # Validation
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        valid_loader_tqdm = tqdm(valid_loader, desc="Validating")
        with torch.no_grad():
            for inputs, labels in valid_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                valid_loader_tqdm.set_postfix(loss=running_loss / len(valid_loader))

        valid_loss=running_loss/len(valid_loader) 
        accuracy = 100 * correct / total
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    test_loss, test_accuracy = test_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%")
    CHECKPOINT_SAVE_PATH = save_checkpoint(model,save_dir,arch,hidden,epochs,learning_rate,train_data)
    MODEL_SAVE_PATH=save_model(model,arch)
    
    

def main():
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset.')
    parser.add_argument('--data_dir', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vit16', choices=['vgg13', 'vit16'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=512, help='Number of hidden units in vgg')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()
    #setting up device
    

    # Load data
    train_loader, valid_loader,test_loader,train_dataset = load_data(args.data_dir,args.arch)

    # Train the model
    train_model(train_loader, valid_loader,test_loader,train_dataset, args.epochs, args.learning_rate,args.hidden, args.arch,args.save_dir,args.gpu)

if __name__ == '__main__':
    main()
