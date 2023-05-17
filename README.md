# Identification of anonymous authors using textual analysis and machine learning

This repository contains the code and resources for the "Identification of anonymous authors using textual analysis and machine learning" project. 

## Description

The goal of this project is to provide an easy to use script to identify how different machine learning algorithms perform in predicting an author of a piece of text.

## Dataset

This project utilised the [All the News](https://www.kaggle.com/datasets/snapcrack/all-the-news) dataset from Kaggle. It is a comprehensive collection of news articles from various sources, spanning a wide range of topics.

To access the dataset, please visit the following link: [All the News Dataset](https://www.kaggle.com/datasets/snapcrack/all-the-news)

## Installation

Please install the modules found in the requirements.txt file to ensure the script runs without issues.

## Usage
Please use the following arguments when running the script either from command line or a python IDE to change how the script runs to your liking.
NOTE: If there are issues running the distilbert model, please change the batch_size argument to a lower value e.g., 4 or 8.

--dataset IF LEFT BLANK = "../Data/articles1.csv". An optional string argument, pass the path to your dataset.

--samples IF LEFT BLANK = 500. An optional integer argument, pass the minimum amount of samples for an author to be accounted for.

--algorithms IF LEFT BLANK = "rf, xgb, mlp, lr, ensemble, distilbert". An optional string argument, pass any/all of these "rf", "xgb", "mlp", "lr", "ensemble", "distilbert".

--batch_size' IF LEFT BLANK = 16. An optional integer argument, pass the batch size to be used for the distilbert model.


By: Szymon Pawlica