import networkx as nx
import hashlib
import numpy as np


# define hash encoding function, as 256 dimensions
def hash_to_vector(attrs):
    attrs_str = "_".join([str(v) for v in attrs.values()])
    hash_digest = hashlib.sha256(attrs_str.encode()).digest()
    # transfer hash output as 0-1 vectors in 256 dimensions
    vector = np.array([bit for byte in hash_digest for bit in format(byte, '08b')], dtype=np.float32)
    return vector


# build graph
G = nx.Graph()

# example data: extract from real data
transactions = [
    {"trans_date_trans_time": "2020/6/21 12:14", "amt": 2.86, "merchant": "fraud_Kirlin and Sons",
     "category": "personal_care",
     "merch_lat": 33.986391, "merch_long": -81.200714, "gender": "M", "street": "351 Darlene Green", "city": "Columbia",
     "state": "SC", "job": "Mechanical engineer", "dob": "1968/3/19", "name": "Jeff Elliott"},
    # add more transactions
]

# add nodes and edges
for transaction in transactions:
    # cardholder nodes
    user_attrs = {
        "gender": transaction["gender"],
        "street": transaction["street"],
        "city": transaction["city"],
        "state": transaction["state"],
        "job": transaction["job"],
        "dob": transaction["dob"],
        "name": transaction["name"]
    }
    user_node = transaction["name"]
    G.add_node(user_node, **user_attrs)

    # merchants nodes
    merchant_attrs = {
        "merchant": transaction["merchant"],
        "category": transaction["category"],
        "merch_lat": transaction["merch_lat"],
        "merch_long": transaction["merch_long"]
    }
    merchant_node = transaction["merchant"]
    G.add_node(merchant_node, **merchant_attrs)

    # transaction edges
    edge_attrs = {
        "trans_date_trans_time": transaction["trans_date_trans_time"],
        "amt": transaction["amt"]
    }
    G.add_edge(user_node, merchant_node, **edge_attrs)

# hash encoding to nodes
node_embeddings = {}
for node, attrs in G.nodes(data=True):
    node_embeddings[node] = hash_to_vector(attrs)

# hash encoding to edges
edge_embeddings = {}
for u, v, attrs in G.edges(data=True):
    edge_embeddings[(u, v)] = hash_to_vector(attrs)

# save nodes and edges embedding
np.savez('node_edge_embeddings.npz', node_embeddings=node_embeddings, edge_embeddings=edge_embeddings)

# save to GML file
nx.write_gml(G, 'encoded_graph.gml')

print("graph is formed and saved to 'encoded_graph.gml'")
