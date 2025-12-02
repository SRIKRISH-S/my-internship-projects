MNIST CNN
---------
A small CNN trained on MNIST digits using TensorFlow.

How to run:
1. Open terminal in VS Code (Terminal → New Terminal).
2. Create & activate virtual environment:
   - Windows:
     python -m venv venv
     venv\Scripts\activate
   - mac/linux:
     python3 -m venv venv
     source venv/bin/activate
3. Install:
   pip install -r requirements.txt
4. Run:
   python train_mnist.py

Model saved to saved_model/mnist_cnn and training_loss.png created.
