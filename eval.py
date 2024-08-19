import torch
import numpy as np
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, roc_auc_score,
                             precision_recall_curve, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, dataloader, threshold=0.5):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch_idx, (merchant_embedding, customer_embedding, transaction_embedding, labels) in enumerate(dataloader):
            # connect all embeddings
            all_embeddings = torch.cat([merchant_embedding, customer_embedding, transaction_embedding], dim=1)
            outputs = model(all_embeddings)

            probabilities = torch.sigmoid(outputs).squeeze()
            predictions = (probabilities > threshold).float()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # evaluation metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    # roc_auc = roc_auc_score(all_labels, all_probabilities)

    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1 Score: {:.4f}".format(f1))
    # print("ROC AUC: {:.4f}".format(roc_auc))
    return 0

# Example usage (assuming you have a trained model and a dataloader):
# metrics = evaluate_model(model, dataloader)
