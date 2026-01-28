# Physics-Guided BiLSTM-Multi-head Attention Model for Flood Forecasting

This repository contains the source code, data, and modelling methods for the paper submitted to *Environmental Modelling & Software*.

The project implements a Physics-Informed Bi-directional LSTM with Multi-Head Attention (BiLSTM-MHA) for real-time flood forecasting. It integrates physical constraints (mass conservation/causality) into the loss function to improve prediction reliability.

## 📂 Repository Structure

* **`BiLSTM_MHA.py`**: The core library file containing:
    * `BiLSTMAttention`: The neural network architecture definition.
    * `PhysicsInformedLoss`: The custom loss function combining MSE and physical constraints.
    * `xaj_mechanism`: Utility function for the Xinanjiang (XAJ) physical model simulation.
* **`Train.py`**: The training script. It handles data loading, feature engineering (XAJ parameters), normalization, and model training using the physics-informed loss[cite: 1].
* **`Test.py`**: The evaluation script. It loads the trained weights (`model_weights.pth`), performs inference on the test set, and calculates performance metrics (NSE, RMSE)[cite: 2].
* **`Data.csv`**: A demonstration dataset containing rainfall and water depth time series.
* `Requirements.txt`**: List of Python dependencies required to run the model.

## 🚀 Quick Start: How to Run the Model

### 1. Environment Setup
Ensure you have Python installed (Python 3.8+ recommended). Install the required dependencies:

```bash
pip install -r requirements.txt

```

*Key dependencies: PyTorch, NumPy, Pandas, Scikit-learn.*

### 2. Training the Model

Run the training script to preprocess the data and train the neural network. This process ensures reproducibility by fixing random seeds.

```bash
python Train.py

```

* **Input:** Reads `Data.csv`.
* **Process:** Trains for 50 epochs (default).
* **Output:**
* Prints training loss logs.
* Saves the best model weights to `model_weights.pth`.



### 3. Testing and Evaluation

Once the model is trained and `model_weights.pth` is generated, run the testing script to examine the model's performance on unseen data.

```bash
python Test.py

```

* **Output:** Displays the evaluation metrics, including:
* **NSE** (Nash-Sutcliffe Efficiency)
* **RMSE** (Root Mean Square Error)



## 📊 Data Description

The model utilizes `Data.csv` for both training and testing. The dataset includes:

**Time**: Timestamp of the record.


**Rainfall(mm/h)**: The primary forcing input.


**True Water Depth (m)**: The target variable (ground truth) used for supervised learning.



Note: The script automatically performs feature engineering to derive Soil Saturation and Runoff using the XAJ mechanism during runtime.

## ⚙️ Model Configuration

Key hyperparameters used in this study (modifiable in `Train.py`):

**Sequence Length**: 6 time steps.


**Hidden Size**: 64 units.


**Physics Weight ()**: 0.2 (Weight for the physical constraint term in Loss).


**Learning Rate**: 0.005.



## 📧 Contact

For any questions regarding the code or data, please contact the corresponding author of the paper.

