# 🧠 Perceptron Classifier in C++

This project contains a straightforward implementation of the **Perceptron** algorithm — a foundational concept in **Artificial Intelligence (AI)** and **Machine Learning (ML)**. The code is written in **C++** and walks you through how a basic linear classifier works from scratch.

---

## 🎯 Purpose

The main purpose of this project is to help learners:

- Understand how **linear classifiers** like the Perceptron operate under the hood.
- Learn how to handle **data loading**, **model training**, **prediction**, and **accuracy evaluation**.
- Explore the Perceptron — a fundamental tool in AI — through an end-to-end, transparent implementation.

---

## 🧩 Features

- Load data from CSV files (`training_data.csv`, `test_data.csv`)
- Parse numerical features and binary class labels (`1` or `-1`)
- Train a Perceptron using the classic learning algorithm
- Predict binary outcomes for test data
- Calculate and display **test accuracy**
- Output learned **weights** and **bias**

---

## 📁 CSV Format

Each row in the CSV file should follow this format:





Perceptron Classifier in C++

This repository provides a simple yet complete implementation of a Perceptron, one of the most fundamental tools in Artificial Intelligence (AI) and Machine Learning. Written entirely in C++, it walks through every step — from reading data to training the model and evaluating its accuracy.

 Purpose : 
The goal of this project is to understand how linear classifiers like Perceptrons operate under the hood — covering:

- Data loading from CSV files
- Vectorized training using the perceptron learning rule
- Binary prediction using learned weights and bias
- Accuracy calculation and model evaluation
- This hands-on approach helps reinforce key AI/ML concepts by implementing them from scratch.

  Part of a University Project for the  AI LAB. 

Build & Run : 
g++ -o perceptron main.cpp
./perceptron

CSV Format
Each row in the CSV should follow the format:

feature1,feature2,...,featureN,label



