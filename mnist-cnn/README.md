# MNIST Handwritten Digit Recognition (CNN Model)

This project uses a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) from the MNIST dataset.

---

## 📌 Overview
The MNIST dataset contains 70,000 images of handwritten digits.  
This model trains a CNN to achieve **~99% test accuracy**.

---

## 🚀 Features
- Trained using TensorFlow/Keras  
- Achieves ~99% accuracy  
- Saved model in `.keras` format  
- Demo script included to visually test predictions  
- Training loss graph saved as `training_loss.png`

---

## 📂 Project Files
```
mnist-cnn/
│── train_mnist.py          # Training script
│── show_mnist_demo.py      # Demo script to show prediction on sample digit
│── training_loss.png       # Loss graph
│── saved_model/
│     └── mnist_cnn.keras   # Saved trained model
```

---

## ▶️ How to Run Training
```
python train_mnist.py
```

## ▶️ How to Run Prediction Demo
```
python show_mnist_demo.py
```

This will:
- Load the saved model  
- Pick a sample MNIST digit  
- Display the image  
- Show **Predicted vs Actual** output

---

## 📈 Example Outputs
- Test Accuracy: ~0.989  
- Training loss graph saved in `training_loss.png`

---

## 👤 Author
**SRIKRISH S**  
GitHub: https://github.com/SRIKRISH-S
