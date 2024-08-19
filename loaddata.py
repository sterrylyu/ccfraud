import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class FraudDataset(Dataset):
    def __init__(self, merchant_embeddings_file, customer_embeddings_file, transaction_embeddings_file, csv_file):
        # load data
        self.merchant_embeddings = np.load(merchant_embeddings_file, allow_pickle=True).item()
        self.customer_embeddings = np.load(customer_embeddings_file, allow_pickle=True).item()
        self.transaction_embeddings = np.load(transaction_embeddings_file, allow_pickle=True).item()

        # load CSV , get trans_id 和label
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.set_index('trans_id')['is_fraud'].to_dict()

        # define dimension
        self.embedding_dim = 256

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取当前索引的 trans_id
        trans_id = self.data.iloc[idx]['trans_id']

        # 获取商家、顾客和交易的嵌入向量字典
        merchant_embedding_dict = self.merchant_embeddings.get(trans_id, {})
        customer_embedding_dict = self.customer_embeddings.get(trans_id, {})
        transaction_embedding_dict = self.transaction_embeddings.get(trans_id, {})

        # 将字典中的所有值转换为 tensor，并按通道拼接
        merchant_embeddings = [torch.tensor(v, dtype=torch.float32).unsqueeze(0) for v in merchant_embedding_dict.values()]
        customer_embeddings = [torch.tensor(v, dtype=torch.float32).unsqueeze(0) for v in customer_embedding_dict.values()]
        transaction_embeddings = [torch.tensor(v, dtype=torch.float32).unsqueeze(0) for v in transaction_embedding_dict.values()]

        # 拼接商家、顾客和交易的嵌入向量
        if merchant_embeddings:
            merchant_embedding = torch.cat(merchant_embeddings, dim=0)
        else:
            merchant_embedding = torch.zeros(1, self.embedding_dim)

        if customer_embeddings:
            customer_embedding = torch.cat(customer_embeddings, dim=0)
        else:
            customer_embedding = torch.zeros(1, self.embedding_dim)

        if transaction_embeddings:
            transaction_embedding = torch.cat(transaction_embeddings, dim=0)
        else:
            transaction_embedding = torch.zeros(1, self.embedding_dim)

        # get binary label
        label = torch.tensor(self.labels.get(trans_id, 0), dtype=torch.float32)

        return merchant_embedding, customer_embedding, transaction_embedding, label

def loadData(merchant_embeddings_file, customer_embeddings_file, transaction_embeddings_file, csv_file, batch_size, shuffle=True):
    dataset = FraudDataset(
        merchant_embeddings_file,
        customer_embeddings_file,
        transaction_embeddings_file,
        csv_file
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
