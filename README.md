# LDI-ML README

Welcome to the LDI-ML repository. This document provides an overview of the notebooks and code included in this project.

## Notebooks

### Trading Analysis
- **trading.ipynb**: This notebook includes functions for loading data, feature engineering, model training, and backtesting strategies using financial data. Key functions include:
  - [load_data()](file:///Users/almazkhalilov/Desktop/uni_sem1_2024/system_optimisation_and_machine_learning/LDI/trading.ipynb#188%2C11-188%2C11): Load stock data from Yahoo Finance.
  - [create_features()](file:///Users/almazkhalilov/Desktop/uni_sem1_2024/system_optimisation_and_machine_learning/LDI/trading.ipynb#190%2C11-190%2C11): Calculate moving averages and RSI for stocks.
  - [train_model()](file:///Users/almazkhalilov/Desktop/uni_sem1_2024/system_optimisation_and_machine_learning/LDI/trading.ipynb#191%2C14-191%2C14): Train a machine learning model on stock data.
  - [backtest_strategy()](file:///Users/almazkhalilov/Desktop/uni_sem1_2024/system_optimisation_and_machine_learning/LDI/trading.ipynb#192%2C6-192%2C6): Backtest trading strategies using Backtrader.

  
```172:192:trading.ipynb
    "def load_data(symbol):\n",
    "    df = yf.download(symbol, period='5y', interval='1d')\n",
    "    return df\n",
    "\n",
    "def save_plot(df, filename='plot.png'):\n",
    "    plt.figure(figsize=(10, 5))\n",
    "    plt.plot(df['Close'], label='Close Price')\n",
    "    plt.title('Stock Closing Prices')\n",
```


### Water Market Analysis
- **WaterMarket.ipynb**: Analyzes water trading data, including preprocessing and machine learning classification to predict water trading prices.
  - Data loading and preprocessing.
  - SVM classification to predict price classes based on trading volume.

  
```244:267:WaterMarket.ipynb
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Feature scaling\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "# Applying nonlinear SVM classification with a polynomial kernel\n",
    "svm_classifier = SVC(kernel='poly')\n",
    "svm_classifier.fit(X_train_scaled, y_train)\n",
    "\n",
    "# Predicting the test set results\n",
    "y_pred = svm_classifier.predict(X_test_scaled)\n",
    "\n",
    "# Evaluating the model\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "conf_matrix = confusion_matrix(y_test, y_pred)\n",
    "class_report = classification_report(y_test, y_pred)\n",
    "\n",
    "print(f\"Accuracy: {accuracy}\")\n",
    "print(\"Confusion Matrix:\")\n",
    "print(conf_matrix)\n",
    "print(\"Classification Report:\")\n",
    "print(class_report)\n",
```


### Basic Models
- **BasicModels1.ipynb**: Demonstrates training of basic machine learning models on sample data, including performance metrics and validation.
  - Training loop for a model with detailed step-by-step loss and validation loss.

  
```499:523:BasicModels1.ipynb
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - loss: 1.4393e-04 - val_loss: 7.8456e-04\n",
      "Epoch 18/100\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 1.5407e-04 - val_loss: 8.1390e-04\n",
      "Epoch 19/100\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 1.3180e-04 - val_loss: 7.3428e-04\n",
      "Epoch 20/100\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 1.2788e-04 - val_loss: 7.2041e-04\n",
      "Epoch 21/100\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 1.3542e-04 - val_loss: 7.2310e-04\n",
      "Epoch 22/100\n",
      "\u001b[1m32/32\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - loss: 1.2395e-04 - val_loss: 7.1636e-04\n",
      "Epoch 23/100\n",
```


## Installation and Setup
Ensure you have Python 3.11.2 installed, as specified in the notebooks. Dependencies include `pandas`, `numpy`, `matplotlib`, `sklearn`, `backtrader`, `yfinance`, and `keras`.

## Running Notebooks
To run the notebooks, ensure you have Jupyter Notebook or JupyterLab installed. Open the desired `.ipynb` file and execute the cells sequentially.

For more details on each notebook, refer to the comments and documentation within each file.