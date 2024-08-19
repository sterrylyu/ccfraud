import numpy as np

# 加载保存的.npy文件
user_embeddings = np.load('user_embeddings.npy', allow_pickle=True).item()
merchant_embeddings = np.load('merchant_embeddings.npy', allow_pickle=True).item()
transaction_embeddings = np.load('transaction_embeddings.npy', allow_pickle=True).item()

# 打印数据的键
print("用户嵌入的键:", list(user_embeddings.keys()))
print("商家嵌入的键:", list(merchant_embeddings.keys()))
print("交易嵌入的键:", list(transaction_embeddings.keys()))

# 打印用户嵌入数据
print("\n用户嵌入示例:")
for key, value in list(user_embeddings.items())[:5]:  # 打印前5个用户
    print(f"  用户: {key}")
    print(f"    嵌入向量示例: {value[:5]}")  # 打印前5个维度

# 打印商家嵌入数据
print("\n商家嵌入示例:")
for key, value in list(merchant_embeddings.items())[:5]:  # 打印前5个商家
    print(f"  商家: {key}")
    print(f"    嵌入向量示例: {value[:5]}")  # 打印前5个维度

# 打印交易嵌入数据
print("\n交易嵌入示例:")
for key, value in list(transaction_embeddings.items())[:5]:  # 打印前5个交易
    print(f"  交易ID: {key}")
    print(f"    嵌入向量示例: {value[:5]}")  # 打印前5个维度
