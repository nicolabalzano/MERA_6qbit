import numpy as np
import tensorflow as tf
import torch
from torch.optim import Adam
from torch.nn import MSELoss, BCELoss
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ==========================================
# 1. DATASET PREPARATION (MNIST)
# ==========================================
def load_and_process_mnist():
    print("Caricamento e preprocessing MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Filtro classi 0 e 1 (Top vs QCD background proxy)
    mask_train = (y_train == 0) | (y_train == 1)
    mask_test = (y_test == 0) | (y_test == 1)
    x_train, y_train = x_train[mask_train], y_train[mask_train]
    x_test, y_test = x_test[mask_test], y_test[mask_test]

    # Riduzione dataset
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:200]
    y_test = y_test[:200]

    # Resize a 6 pixel (3x2)
    x_train_resized = tf.image.resize(x_train[..., np.newaxis], (3, 2)).numpy().reshape(-1, 6)
    x_test_resized = tf.image.resize(x_test[..., np.newaxis], (3, 2)).numpy().reshape(-1, 6)

    # Normalizzazione [0, pi]
    x_train_norm = (x_train_resized / 255.0) * np.pi
    x_test_norm = (x_test_resized / 255.0) * np.pi

    # Conversione in Tensor PyTorch
    X_train_torch = torch.tensor(x_train_norm, dtype=torch.float32)
    Y_train_torch = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test_torch = torch.tensor(x_test_norm, dtype=torch.float32)
    Y_test_torch = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch

X_train, Y_train, X_test, Y_test = load_and_process_mnist()

# ==========================================
# 2. DEFINIZIONE CIRCUITO PARAMETRICO (MERA-like)
# ==========================================
n_qubits = 6

def create_qnn():
    qc = QuantumCircuit(n_qubits)
    
    # Parametri Input (Dati)
    input_params = ParameterVector('x', n_qubits)
    # Parametri Pesi (Training) - 9 blocchi * 2 parametri = 18
    weight_params = ParameterVector('theta', 18)
    
    for i in range(n_qubits):
        qc.ry(input_params[i], i)
        
    idx = 0
    
    def apply_block(wires):
        nonlocal idx
        qc.ry(weight_params[idx], wires[0])
        qc.ry(weight_params[idx+1], wires[1])
        qc.cx(wires[0], wires[1])
        idx += 2

    apply_block([1, 2]) # Wires 1-2
    apply_block([3, 4]) # Wires 3-4
    
    apply_block([0, 1]) # Wires 0-1
    apply_block([2, 3]) # Wires 2-3
    apply_block([4, 5]) # Wires 4-5
    
    apply_block([1, 4]) # Wires 1-4
    
    apply_block([1, 2]) # Wires 1-2
    apply_block([3, 4]) # Wires 3-4
    
    apply_block([1, 4]) # Wires 1-4
    
    # -- Observables --
    observable = SparsePauliOp.from_list([("IZIIII", 1)])
    
    # Definizione QNN
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=input_params,
        weight_params=weight_params,
        observables=observable
    )
    
    return qnn, qc

qnn, qc = create_qnn()

# Stampa del circuito
print(qc.draw())
qc.draw(output='mpl', filename='mera_circuit.png')
plt.show()

# ==========================================
# 3. MODELLO PYTORCH E TRAINING (Adam)
# ==========================================

class QuantumMERAClassifier(torch.nn.Module):
    def __init__(self, qnn):
        super().__init__()
        self.qnn_layer = TorchConnector(qnn, initial_weights=np.random.randn(qnn.num_weights) * 0.1)
        
    def forward(self, x):
        exp_val = self.qnn_layer(x)
        
        prob = (1 - exp_val) / 2 
        return prob

model = QuantumMERAClassifier(qnn)

optimizer = Adam(model.parameters(), lr=0.01)
criterion = BCELoss()

# Training Loop
epochs = 15
batch_size = 20
dataset = torch.utils.data.TensorDataset(X_train, Y_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("\nInizio Training")
loss_history = []

model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()           # Reset gradienti
        output = model(batch_x)         # Forward pass
        loss = criterion(output, batch_y) # Calcolo Loss
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update dei pesi
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    
    # Calcolo accuracy sul train
    with torch.no_grad():
        train_preds_prob = model(X_train)
        train_preds = (train_preds_prob > 0.5).float()
        acc = accuracy_score(Y_train, train_preds)
        
    print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")

# ==========================================
# 4. TEST FINALE
# ==========================================
model.eval()
with torch.no_grad():
    test_probs = model(X_test)
    test_preds = (test_probs > 0.5).float()
    test_acc = accuracy_score(Y_test, test_preds)

print(f"\nTest Accuracy: {test_acc:.4f}")
