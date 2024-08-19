import pandas as pd
import numpy as np
import hashlib
import os

def hash_to_vector(attrs):
    attrs_str = "_".join([str(v) for v in attrs])
    hash_digest = hashlib.sha256(attrs_str.encode()).digest()
    vector = np.array([bit for byte in hash_digest for bit in format(byte, '08b')], dtype=np.float32)
    return vector

def generate_embeddings(csv_file, output_dir):
    # make sure existing catalog for output
    os.makedirs(output_dir, exist_ok=True)

    # dictionary of initialization
    merchant_embeddings = {}
    customer_embeddings = {}
    transaction_embeddings = {}
    labels = {}

    # if the file exist, load existing data
    if os.path.exists(os.path.join(output_dir, 'merchant_embeddings.npy')):
        merchant_embeddings = np.load(os.path.join(output_dir, 'merchant_embeddings.npy'), allow_pickle=True).item()
    if os.path.exists(os.path.join(output_dir, 'customer_embeddings.npy')):
        customer_embeddings = np.load(os.path.join(output_dir, 'customer_embeddings.npy'), allow_pickle=True).item()
    if os.path.exists(os.path.join(output_dir, 'transaction_embeddings.npy')):
        transaction_embeddings = np.load(os.path.join(output_dir, 'transaction_embeddings.npy'), allow_pickle=True).item()
    if os.path.exists(os.path.join(output_dir, 'labels.npy')):
        labels = np.load(os.path.join(output_dir, 'labels.npy'), allow_pickle=True).item()

    # load data
    data = pd.read_csv(csv_file)

    for index, row in data.iterrows():
        trans_id = row['trans_id']
        merchant_id = trans_id  # Assuming trans_id is used as merchant_id
        customer_id = trans_id  # Assuming trans_id is used as customer_id

        # merchant node embeddings
        merchant_embeddings[trans_id] = {
            'merchant': hash_to_vector([merchant_id]),
            'category': hash_to_vector([row['category']]),
            'merch_lat': hash_to_vector([row['merch_lat']]),
            'merch_long': hash_to_vector([row['merch_long']])
        }

        # cardholder node embeddings
        customer_embeddings[trans_id] = {
            'gender': hash_to_vector([row['gender']]),
            'street': hash_to_vector([row['street']]),
            'city': hash_to_vector([row['city']]),
            'state': hash_to_vector([row['state']]),
            'job': hash_to_vector([row['job']]),
            'dob': hash_to_vector([row['dob']])
        }

        # transaction edge embeddings
        transaction_embeddings[trans_id] = {
            'transaction_time': hash_to_vector([row['trans_date_trans_time']]),
            'amount': hash_to_vector([row['amt']])
        }

        # 记录二分类标签
        labels[trans_id] = row['is_fraud']

    # 保存嵌入数据和标签
    np.save(os.path.join(output_dir, 'merchant_embeddings.npy'), merchant_embeddings)
    np.save(os.path.join(output_dir, 'customer_embeddings.npy'), customer_embeddings)
    np.save(os.path.join(output_dir, 'transaction_embeddings.npy'), transaction_embeddings)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)

    print("嵌入数据和标签已生成并保存到指定目录")

if __name__ == '__main__':
    # 这里可以替换成实际的数据文件路径和输出目录
    csv_file = 'data_train.csv'
    output_dir = 'preprocessed_data'
    generate_embeddings(csv_file, output_dir)
