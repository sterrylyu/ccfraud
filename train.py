import torch
import torch.nn as nn
import torch.optim as optim
from utils.loaddata import loadData
from model.attention import AttentionModel


def train(model, dataloader, criterion, optimizer, num_epochs=500):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (merchant_embedding, customer_embedding, transaction_embedding, labels) in enumerate(dataloader):
            # Forward pass
            # connect all embedding channels
            # combine all channels
            all_embeddings = torch.cat([merchant_embedding, customer_embedding, transaction_embedding], dim=1)
            outputs = model(all_embeddings)
            # print(outputs, labels)
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute metrics
            epoch_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item()}')

        epoch_accuracy = correct_predictions / total_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss / len(dataloader)}, Accuracy: {epoch_accuracy}')


if __name__ == '__main__':
    # data file path
    data_dir = 'preprocessed_data'
    merchant_embeddings_file = data_dir + '/merchant_embeddings.npy'
    customer_embeddings_file = data_dir + '/customer_embeddings.npy'
    transaction_embeddings_file = data_dir + '/transaction_embeddings.npy'
    labels_file = data_dir + '/labels.npy'
    csv_file = 'data_simple.csv.csv'

    # load data
    dataloader = loadData(
        merchant_embeddings_file,
        customer_embeddings_file,
        transaction_embeddings_file,
        csv_file,
        batch_size=4,
        shuffle=True
    )

    # model、loss function and optimizer
    input_dim = 256  # 256 dimension
    model = AttentionModel(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train model
    train(model, dataloader, criterion, optimizer, num_epochs=500)

    # save model
    torch.save(model.state_dict(), 'attention_model.pth')
