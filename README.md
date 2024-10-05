# Vision Transformer (ViT-B-16) Flower Classifier Image Classifier

## Overview

This project focuses on utilizing the Vision Transformer (ViT-B/16) model to classify 102 different flower species. By employing transfer learning, the project efficiently builds on a pre-trained ViT model and tailors it for the flower classification task. The head of the model has been modified to suit the dataset, and the project is optimized for quick training and high accuracy.

### Features

- **Pre-trained ViT Models**: Leverages the Vision Transformer (ViT-B/16) for top-tier image classification performance.
- **Fast Training**: Optimized to run with a batch size of 32, 5 epochs, and a learning rate of 1e-3, making training time around 10 minutes.
- **GPU Support**: If a GPU is available, training can be done even faster.
- **Image Preprocessing**: Preprocessing utilities ensure compatibility with the ViT architecture.
- **Label Mapping**:A JSON file can be used to map numerical class labels to their corresponding flower names.
- **Model Saving and Loading**: Save trained models for future use and load them easily for predictions.

## Getting Started

### Prerequisites

- Python
- Pip for managing dependencies
- use PyTorch v1.12+ and torchvision v0.14+
- Required Python libraries: `torch`, `torchvision`, `PIL`, `json`, `collections`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Aasthajain02/AWS_Flower_ImageClassifier.git
   cd AWS_Flower_ImageClassifier
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Training the Model**:
   To train your custom image classifier, run the following command:

   ```bash
   python train.py --data_dir <path_to_training_data> --save_dir <path_to_save_model> --model vit_b_16 batch_size 32 --epochs 5 --learning_rate 1e-3

   ```
   --model vit_b_16 specifies the Vision Transformer model, with a head modified for flower classification.
Training uses a batch size of 32 and runs for 5 epochs with a learning rate of 1e-3.
2. **Predicting Images**:
   To make predictions on new images, use the following command:

   ```bash
   python predict.py <image_path> <checkpoint_path> --top_k <number_of_top_predictions> --category_names <path_to_category_names.json> --gpu
   ```

   - Replace <image_path> with the path to the image you want to classify.
   - Replace <checkpoint_path> with the path to your saved model.
   - The --top_k option shows the top predicted classes (e.g., top 5).
   - Use the --category_names option to map class indices to actual flower names.

### Example

```bash
# Example to predict an image
python predict.py flower_data/test/image_05100.jpg saved_models/vit_b_16_checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu

```

## Project Overview

This project is part of the Udacity Nanodegree and is centered around the ViT-B/16 model for classifying 102 types of flowers. The custom classifier head allows it to fit the specific dataset. Using a batch size of 32, 5 epochs, and a learning rate of 1e-3, the model achieved over 90% accuracy on the test set and 82% on the validation set, all within 10 minutes of training.
