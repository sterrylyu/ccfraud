# Train.py

## Overview
`train.py` is a script designed for training a binary classification model using an attention-based model (`AttentionModel`). It processes three types of embeddings: merchant, customer, and transaction embeddings, concatenates them, and predicts the label of a given transaction.

## File Structure
- `train.py`: The main training script, including the training loop and model evaluation.
- `utils/loaddata.py`: Contains the `loadData` function for loading embeddings and labels.
- `model/attention.py`: Defines the attention-based model `AttentionModel`.

## Usage

### Data Preparation
Before running `train.py`, ensure you have the following files ready:
1. **Merchant Embeddings:** The `merchant_embeddings.npy` file containing merchant embeddings.
2. **Customer Embeddings:** The `customer_embeddings.npy` file containing customer embeddings.
3. **Transaction Embeddings:** The `transaction_embeddings.npy` file containing transaction embeddings.
4. **Labels:** The `labels.npy` file containing the binary labels for the transactions.
5. **CSV File:** The `data_simple.csv` file containing metadata or other auxiliary information for the dataset (if applicable).

Place these files in the `preprocessed_data` directory, or modify the `data_dir` path in the code according to your setup.

### Running the Training Script
To start training the model, run the following command in the terminal:

```bash
python train.py
```

### Parameter Configuration
- **Data Loading Parameters:**
  - `batch_size=4`: Number of samples per training batch.
  - `shuffle=True`: Shuffle the data while loading it.

- **Model, Loss Function, and Optimizer:**
  - `input_dim=256`: Input embedding dimension. Adjust this according to the dimensionality of your data.
  - `criterion=nn.BCELoss()`: Use binary cross-entropy loss for the classification task.
  - `optimizer=optim.Adam(model.parameters(), lr=0.001)`: Use the Adam optimizer with a learning rate of `0.001`.

- **Training Parameters:**
  - `num_epochs=500`: Number of training epochs. Adjust this based on your dataset size and model complexity.

### Training Output
During training, the script will print the loss for every 10 batches and the average loss and accuracy at the end of each epoch.

### Saving the Model
To save the trained model for later inference or further testing, it is recommended to add a model-saving functionality in the training script. You can add the following code at the end of training:

```python
torch.save(model.state_dict(), 'attention_model.pth')
```

This command will save the trained model parameters to a `attention_model.pth` file, which can be loaded later for inference or further training.
