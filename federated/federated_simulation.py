"""
federated/federated_simulation.py
Simulates federated averaging across 3 hospital nodes.
Each node trains on a local partition; global model is FedAvg'd.
No data leaves each node — only model weights are aggregated.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from models.model import build_model

DEVICE       = "cpu"
ARTIFACT_DIR = "artifacts"
NUM_NODES    = 3
FL_ROUNDS    = 5
LOCAL_EPOCHS = 2
BATCH_SIZE   = 32
LR           = 5e-4


def partition_data(X, y, num_nodes: int):
    """Split dataset into num_nodes non-overlapping shards (IID)."""
    n = len(X)
    idx = np.random.permutation(n)
    shards = np.array_split(idx, num_nodes)
    return [(X[s], y[s]) for s in shards]


def local_train(global_state_dict: dict, X_node: np.ndarray, y_node: np.ndarray,
                node_id: int) -> dict:
    """Train one node for LOCAL_EPOCHS and return updated weights."""
    input_size = X_node.shape[2]
    model = build_model(input_size, DEVICE)
    model.load_state_dict(copy.deepcopy(global_state_dict))
    model.train()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_node), torch.from_numpy(y_node)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    for epoch in range(LOCAL_EPOCHS):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            prob, _ = model(xb)
            loss = criterion(prob, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    print(f"    Node {node_id}: loss={epoch_loss/len(loader):.4f} | samples={len(y_node)}")
    return model.state_dict()


def federated_average(state_dicts: list, weights: list) -> dict:
    """Weighted FedAvg: average model weights proportional to local dataset size."""
    total = sum(weights)
    avg_state = copy.deepcopy(state_dicts[0])

    for key in avg_state:
        avg_state[key] = sum(
            state_dicts[i][key].float() * (weights[i] / total)
            for i in range(len(state_dicts))
        )
    return avg_state


def run_federated_simulation():
    print("\n" + "═" * 55)
    print("  Aegis-Omni — Federated Learning Simulation")
    print(f"  Nodes: {NUM_NODES}  |  Rounds: {FL_ROUNDS}  |  Local epochs: {LOCAL_EPOCHS}")
    print("═" * 55)

    # Load global dataset
    X_train = np.load(f"{ARTIFACT_DIR}/X_train.npy")
    y_train = np.load(f"{ARTIFACT_DIR}/y_train.npy")
    X_test  = np.load(f"{ARTIFACT_DIR}/X_test.npy")
    y_test  = np.load(f"{ARTIFACT_DIR}/y_test.npy")

    # Partition across nodes
    partitions = partition_data(X_train, y_train, NUM_NODES)
    node_sizes = [len(p[1]) for p in partitions]
    print(f"  Node sizes: {node_sizes}\n")

    # Initialise global model
    input_size    = X_train.shape[2]
    global_model  = build_model(input_size, DEVICE)
    global_state  = global_model.state_dict()

    for round_num in range(1, FL_ROUNDS + 1):
        print(f"  ── Round {round_num} ──────────────────────────────")
        local_states = []

        for node_id, (X_node, y_node) in enumerate(partitions):
            local_state = local_train(global_state, X_node, y_node, node_id)
            local_states.append(local_state)

        # Aggregate
        global_state = federated_average(local_states, node_sizes)
        global_model.load_state_dict(global_state)

        # Evaluate global model on held-out test set
        global_model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_test), 256):
                xb = torch.from_numpy(X_test[i:i+256])
                prob, _ = global_model(xb)
                preds.append(prob.numpy())
        preds = np.concatenate(preds)
        auroc = roc_auc_score(y_test, preds)
        print(f"  → Global AUROC after round {round_num}: {auroc:.4f}\n")

    # Save federated checkpoint
    torch.save(global_state, f"{ARTIFACT_DIR}/federated_model.pt")
    print(f"  [FL] Federated model saved → {ARTIFACT_DIR}/federated_model.pt")
    return global_model


if __name__ == "__main__":
    run_federated_simulation()
