import torch
from utils.loaddata import loadData
from model.attention import AttentionModel
from eval import evaluate_model

def infer(model, dataloader):
    model.eval()  # switch to evaluation

    all_predictions = []
    all_labels = []

    with torch.no_grad():  # close gradient to save storage
        for batch_idx, (merchant_embedding, customer_embedding, transaction_embedding, labels) in enumerate(dataloader):
            # connect all embedding methods
            all_embeddings = torch.cat([merchant_embedding, customer_embedding, transaction_embedding], dim=1)
            outputs = model(all_embeddings)

            predicted = (outputs.squeeze() > 0.5).float()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_predictions, all_labels

if __name__ == '__main__':
    # data file path
    data_dir = 'preprocessed_data'
    merchant_embeddings_file = data_dir + '/merchant_embeddings.npy'
    customer_embeddings_file = data_dir + '/customer_embeddings.npy'
    transaction_embeddings_file = data_dir + '/transaction_embeddings.npy'
    labels_file = data_dir + '/labels.npy'
    csv_file = 'data_simple.csv'

    # load data
    dataloader = loadData(
        merchant_embeddings_file,
        customer_embeddings_file,
        transaction_embeddings_file,
        csv_file,
        batch_size=4,
        shuffle=False  # no need to shuffle during prediction
    )

    # load model after training
    input_dim = 256  # 256 dimensions
    model = AttentionModel(input_dim)
    model.load_state_dict(torch.load('attention_model.pth'))  # save in 'attention_model.pth'

    # predictions
    predictions, labels = infer(model, dataloader)

    # evaluate the result for predictions
    correct_predictions = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct_predictions / len(labels)
    print(f'Inference Accuracy: {accuracy:.4f}')

    # evaluate model
    metrics = evaluate_model(model, dataloader)
