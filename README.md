# Vegetable Recognition AI

This project is an AI-powered application that recognizes various vegetables from images using a Convolutional Neural Network (CNN) based on ResNet-50.

## Supported Classes

The model currently recognizes the following vegetables:

- Potato 
- Broccoli  
- Cucumber 
- Tomato  
- Carrot 

## Features

- Trains a CNN to classify vegetable images.
- Saves the trained model for reuse without retraining.
- Loads the saved model automatically if available.
- Provides an interactive web interface to upload images and get predictions.
- Outputs confidence scores with predictions.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- Pillow
- gradio

Install dependencies with:

```bash
pip install torch torchvision numpy matplotlib pillow gradio
