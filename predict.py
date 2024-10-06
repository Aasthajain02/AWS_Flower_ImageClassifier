import os
import copy
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn, optim
import torchmetrics
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
#from torchmetrics import MulticlassAccuracy, MulticlassF1Score  # Ensure torchmetrics is installed
from train import create_model


import collections
import json
# def get_label_map(cat_name_file):
#     # Open and load the JSON file
#     with open(cat_name_file, 'r') as file:
#         data = json.load(file)

#     # Convert string keys to integers and store them in a new dictionary
#     category_mapping = {int(key): value for key, value in data.items()}
#     print(category_mapping)
    
#     return category_mapping


def process_image(image_path):
    """Adjusts a PIL image for input into a PyTorch model, 
       returning a Numpy array.
    """
    
    # Load the image using PIL
    image = Image.open(image_path)
    
    # Define the transformations: resize, crop, convert to tensor, and normalize
    transformation_pipeline = transforms.Compose([
        transforms.Resize(size=(224,224),interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Apply the transformations
    image_tensor = transformation_pipeline(image)
    
    return image_tensor

def load_checkpoint(checkpoint_save_path, device) -> dict:
    '''
    Function for loading checkpoint dictionary from given path of the saved checkpoint.

    Args:
        checkpoint_save_path: Path to the general checkpoint .pth file
        gpu: Boolean indicating whether to use GPU

    Returns:
        The checkpoint dictionary (dict)
    '''
    # Load the checkpoint dictionary
    checkpoint_dict = torch.load(checkpoint_save_path, map_location=device,weights_only=False)
    
    return checkpoint_dict

def load_model(checkpoint_dict: dict,
               hidden_layers: tuple,
               output_layer: int,
               device_trained_on: torch.device,
               device: torch.device) -> nn.Module:

    # Extract architecture and parameters from the checkpoint
    arch ="vit16"
   # num_classes = checkpoint_dict['num_classes']
    print(f"Loading model architecture: {arch}")
    print(f"Hidden layers: {hidden_layers}")
    #print(f"Number of classes: {checkpoint_dict['num_classes']}")
    # Build the model based on architecture
    if arch == "vgg13":
        model = models.vgg13(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(25088, 512),  # Adjust input size based on VGG output
            nn.ReLU(),
            nn.Linear(512,102)
        )
    elif arch == "vit16":
        HIDDEN_LAYER_1 = 256
        HIDDEN_LAYER_2 = 128
        model = models.vit_b_16(weights=None)
        model.heads = nn.Sequential(
            nn.Linear(768, HIDDEN_LAYER_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_1, HIDDEN_LAYER_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_2,102)
        )
    else:
        raise ValueError('Invalid architecture. Choose vgg13 or vit16.')
    
    for param in model.parameters():
        param.requires_grad = False

    # loading model state_dict
    if str(device_trained_on) in ("cuda", "cuda:0"):
        # saved on GPU, loading on CPU case
        if str(device) == "cpu":
            model.load_state_dict(checkpoint_dict['model_state_dict'])
        # saved on GPU, loading on GPU case
        elif str(device) in ("cuda", "cuda:0"):
            model.load_state_dict(checkpoint_dict['model_state_dict'])
    else:
        # saved on CPU, loading on CPU case
        if str(device) == "cpu":
            model.load_state_dict(checkpoint_dict['model_state_dict'])
        # saved on CPU, loading on GPU case
        elif str(device) in ("cuda", "cuda:0"):
            model.load_state_dict(checkpoint_dict['model_state_dict'])
            
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    model.class_to_idx = checkpoint_dict['class_to_idx']
    # moving model to device and switching to eval mode
    model.to(device)
    model.eval()
    return model



def predict_and_plot_topk(model: nn.Module,
                          class_to_idx:dict,
                        cat_name_file:json,
                          CLASSES: list,
                          image_path: str,
                          device: torch.device,
                          topk: int = 5):

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = process_image(image_path)
    input_batch = input_tensor.unsqueeze(0).to(device)
    print(f"Processed image shape: {input_tensor.shape}")
    # Moving model to device and switching to eval mode
    model.to(device)
    model.eval()

    # Make predictions
    with torch.inference_mode():
        output = model(input_batch)

    # Convert the output to probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top-k class indices and probabilities
    topk_probs, topk_indices = torch.topk(probabilities, topk)
    topk_probs_np = topk_probs.numpy(force=True)
    topk_indices_np = topk_indices.numpy(force=True)
    
    with open(cat_name_file, 'r') as f:
        cat_to_name = json.load(f)
    
    topk_classes = [CLASSES[index] for index in topk_indices_np]
    print("actual topk classes",topk_classes)
    labels = [cat_to_name[str(cls)] for cls in topk_classes]
    print("actual topk NAMES",labels)
    # Get the top-k class names using class_to_idx and cat_to_name
    #topk_class_names = [cat_to_name[str(class_to_idx[i])] for i in topk_indices_np]
    # Convert tensor to numpy array for plotting
    probs_np = probabilities.numpy(force=True)
    # print(f"Output probabilities: {probabilities}")  # Add this line
    # print(f"Top-k indices: {topk_indices}")  # Add this line
    # print(f"Top-k probabilities: {topk_probs}")
    # Create a horizontal bar graph
    plt.figure(figsize=(10, 6))

    # Plot the image
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')

    # Plot the top-k classes
    plt.subplot(2, 1, 2)
    #plt.barh([class_list[i] for i in topk_indices_np], topk_probs_np, color='blue')
    plt.barh(labels, topk_probs_np, color='blue')
    plt.xlabel('Predicted Probability')
    plt.title(f'Top-{topk} Predicted Classes')

    plt.tight_layout()
    plt.show()



# Main function
def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained model.')

    # Add arguments
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint_path', type=str,default="checkpoint.pth", help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to the category names mapping file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    # Parse the arguments
    args = parser.parse_args()

    # Now you can use args.image_path, args.checkpoint_path, args.top_k, args.category_names, and args.gpu in your code
    print(f'Image Path: {args.image_path}')
    print(f'Checkpoint Path: {args.checkpoint_path}')
    print(f'Top K: {args.top_k}')
    print(f'Category Names Path: {args.category_names}')
    print(f'Use GPU: {args.gpu}')

    # Load data

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    # Initialize model
    test_transforms = transforms.Compose([transforms.Resize(size=(224,224),
                                                         interpolation=transforms.InterpolationMode.BILINEAR),
                                       transforms.RandomRotation(degrees=(-15,-15),
                                                                 interpolation = transforms.InterpolationMode.BILINEAR),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.CenterCrop(size=(224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
    CHKPT_DICT = load_checkpoint(args.checkpoint_path,device)
    
    class_to_idx=CHKPT_DICT["class_to_idx"]  # Note the lowercase 'class_to_idx'
    CLASSES = [int(class_name) for class_name in class_to_idx.keys()]
    print(CLASSES)
    #CLASSES = [class_name for class_name in class_to_idx.values()]
    new_model = load_model(CHKPT_DICT,hidden_layers = CHKPT_DICT["hidden_layers"],output_layer = CHKPT_DICT["output_layer"],device_trained_on = CHKPT_DICT["device_trained_on"],device=device)
    if new_model is None:
            print("Failed to load model. Exiting...")
    # Assuming you have a PyTorch model 'new_model', a list of class names extracted using dataset_name.classes, and an image file path
    predict_and_plot_topk(new_model,class_to_idx ,args.category_names,CLASSES,args.image_path,device,args.top_k)
    

if __name__ == "__main__":
    main()
