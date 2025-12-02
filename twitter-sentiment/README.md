Twitter Sentiment
-----------------
Train a simple TF-IDF + Logistic Regression model on the tweet_eval sentiment dataset.

How to run:
1. Open terminal in VS Code inside this folder.
2. Create & activate virtual environment (see MNIST README).
3. Install:
   pip install -r requirements.txt
4. Run:
   python train_sentiment.py

If dataset download fails due to internet, prepare a CSV with 'text' and 'label' columns and modify the script to read it with pandas.
