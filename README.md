# AWS_Flower_Image_Classifier
The Custom Flower Image Classifier project is designed to help users create their own image classification models using transfer learning and pre-trained deep learning models. This project allows for efficient, high-accuracy image classification with minimal training time, making it accessible to both beginners and advanced users.

Project Features
Custom Model Training: Train your custom flower classifier using a pre-trained model and your dataset.
Pre-trained Models: Leverage the Vision Transformer (ViT) model for faster training and high accuracy.
Model Architecture Exploration: With torchinfo, navigate and inspect pretrained model layers, enabling users to understand the model's structure.
User-Friendly Interface: Simple interface for training, predictions, and model evaluation.
GPU Support: Option to use GPU for accelerated training, reducing training time to 10 minutes.
Image Preprocessing: Inbuilt image preprocessing to ensure compatibility with the model architecture.
Class Label Mapping: Easily interpret predictions with a JSON file mapping class indices to flower names.
Model Saving & Loading: Save trained models and load them for future use in predictions.
Getting Started
Prerequisites
Python 
Pip (Python package installer)
Required Python libraries: torch, torchvision, PIL, json, torchinfo,torchmetrics
Installation
Clone the repository:

git clone https://github.com/Aasthajain02/AWS_Flower_ImageClassifier.git
cd AWS_Flower_ImageClassifier
Install the required libraries:

pip install -r requirements.txt
Usage
Training the Model
To train your image classifier, use the following command:


python train.py --data_dir <path_to_training_data> --save_dir <path_to_save_model>
Predicting Images
To classify an image, use:


python predict.py <image_path> <checkpoint_path> --top_k <number_of_top_predictions> --category_names <path_to_category_names.json> --gpu
Example
To predict the top 5 classes for a sample flower image:


python predict.py flower_data/test/image_05100.jpg saved_models/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
About the Project
The Custom Flower Image Classifier was built as part of a final project for the Udacity Nanodegree. It is optimized for quick training (only 10 minutes) and achieves 90% accuracy on the test dataset.
