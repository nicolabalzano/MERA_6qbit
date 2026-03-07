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
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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

    # Riduzione dataset: 1000 train, 200 validation, 200 test
    x_val = x_train[1000:1200]
    y_val = y_train[1000:1200]
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:200]
    y_test = y_test[:200]

    # Resize a 6 pixel (3x2)
    x_train_resized = tf.image.resize(x_train[..., np.newaxis], (3, 2)).numpy().reshape(-1, 6)
    x_val_resized = tf.image.resize(x_val[..., np.newaxis], (3, 2)).numpy().reshape(-1, 6)
    x_test_resized = tf.image.resize(x_test[..., np.newaxis], (3, 2)).numpy().reshape(-1, 6)

    # Normalizzazione [0, pi]
    x_train_norm = (x_train_resized / 255.0) * np.pi
    x_val_norm = (x_val_resized / 255.0) * np.pi
    x_test_norm = (x_test_resized / 255.0) * np.pi

    # Conversione in Tensor PyTorch
    X_train_torch = torch.tensor(x_train_norm, dtype=torch.float32)
    Y_train_torch = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_torch = torch.tensor(x_val_norm, dtype=torch.float32)
    Y_val_torch = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    X_test_torch = torch.tensor(x_test_norm, dtype=torch.float32)
    Y_test_torch = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    return X_train_torch, Y_train_torch, X_val_torch, Y_val_torch, X_test_torch, Y_test_torch

X_train, Y_train, X_val, Y_val, X_test, Y_test = load_and_process_mnist()

# ==========================================
# 2. DEFINIZIONE CIRCUITO PARAMETRICO (MERA-like)
# ==========================================
n_qubits = 6

def create_qnn(variant='mera'):
    qc = QuantumCircuit(n_qubits)
    
    # Parametri Input (Dati)
    input_params = ParameterVector('x', n_qubits)
    
    # Calcola quanti parametri per blocco servono
    if variant == 'mera' or variant == 'RyRyCNOT':
        params_per_block = 2
    elif variant in ['RyCNOTRy', '(RyRy)CNOT(RyRx)', 'RxRzCNOT']:
        params_per_block = 4
    else:
        raise ValueError(f"Variante '{variant}' non supportata")
        
    num_blocks = 9
    # Parametri Pesi (Training)
    weight_params = ParameterVector('theta', num_blocks * params_per_block)
    
    for i in range(n_qubits):
        qc.ry(input_params[i], i)
        
    idx = 0
    
    def apply_block(wires, up_or_down=1): # 1 one if the connection is from the top to bottom to define the pointing of the CNOT
        nonlocal idx
        if variant == 'mera' or variant == 'RyRyCNOT':
            qc.ry(weight_params[idx], wires[0])
            qc.ry(weight_params[idx+1], wires[1])
            if up_or_down:
                qc.cx(wires[0], wires[1])
            else:
                qc.cx(wires[1], wires[0])
            idx += 2
        elif variant == 'RyCNOTRy':
            qc.ry(weight_params[idx], wires[0])
            qc.ry(weight_params[idx+1], wires[1])
            if up_or_down:                qc.cx(wires[0], wires[1])
            else:
                qc.cx(wires[1], wires[0])
            qc.ry(weight_params[idx+2], wires[0])
            qc.ry(weight_params[idx+3], wires[1])
            idx += 4
        elif variant == '(RyRy)CNOT(RyRx)':
            qc.ry(weight_params[idx], wires[0])
            qc.ry(weight_params[idx+1], wires[1])
            if up_or_down:
                qc.cx(wires[0], wires[1])
                qc.rx(weight_params[idx+2], wires[1])
                qc.ry(weight_params[idx+3], wires[0])
            else:
                qc.cx(wires[1], wires[0])
                qc.ry(weight_params[idx+2], wires[1])
                qc.rx(weight_params[idx+3], wires[0])
            idx += 4
        elif variant == 'RxRzCNOT':
            qc.rx(weight_params[idx], wires[0])
            qc.rz(weight_params[idx+1], wires[0])
            qc.rx(weight_params[idx+2], wires[1])
            qc.rz(weight_params[idx+3], wires[1])
            if up_or_down:
                qc.cx(wires[0], wires[1])
            else:
                qc.cx(wires[1], wires[0])
            idx += 4

    apply_block([1, 2]) # Wires 1-2
    apply_block([3, 4], 0) # Wires 3-4
    
    apply_block([0, 1]) # Wires 0-1
    apply_block([2, 3]) # Wires 2-3
    apply_block([4, 5], 0) # Wires 4-5
    
    apply_block([1, 4], 0) # Wires 1-4
    
    apply_block([1, 2], 0) # Wires 1-2
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

import os
import copy

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): Quante epoche aspettare dopo l'ultimo miglioramento.
            min_delta (float): Miglioramento minimo per essere considerato tale.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Salva una copia profonda dei pesi migliori
            self.best_model_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            print(f"--- EarlyStopping counter: {self.counter} di {self.patience} ---")
            if self.counter >= self.patience:
                self.early_stop = True

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


if __name__ == "__main__":
    print("===== VQC Architecture: " + architecture + " =====")
    #['RyCNOTRy', '(RyRy)CNOT(RyRx)', 'RxRzCNOT']
    architecture = "RyCNOTRy"
    qnn, qc = create_qnn(architecture)
    model = QuantumMERAClassifier(qnn)

    # Create output directory
    out_dir = f"results_{architecture}"
    os.makedirs(out_dir, exist_ok=True)

    # Stampa del circuito
    print(qc.draw())
    qc.draw(output='mpl', filename=os.path.join(out_dir, f'mera_circuit_{architecture}.png'))
    # Removed plt.show() which blocks execution

    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = BCELoss()

    # Training Loop
    epochs = 50  # Aumentato perché l'early stopping fermerà il training
    batch_size = 20
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("\nInizio Training")
    loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # Inizializzazione EarlyStopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

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
        
        # Calcolo validation loss e accuracy
        model.eval()
        with torch.no_grad():
            # Validation
            val_preds_prob = model(X_val)
            val_loss = criterion(val_preds_prob, Y_val).item()
            val_loss_history.append(val_loss)
            
            # Train accuracy
            train_preds_prob = model(X_train)
            train_preds = (train_preds_prob > 0.5).float()
            train_acc = accuracy_score(Y_train, train_preds)
            train_acc_history.append(train_acc)
            
            # Validation accuracy
            val_preds = (val_preds_prob > 0.5).float()
            val_acc = accuracy_score(Y_val, val_preds)
            val_acc_history.append(val_acc)
        model.train()
            
        print(f"Epoch {epoch+1:2d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early Stopping check sulla validation loss
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\nEarly stopping attivato all'epoca {epoch+1}!")
            break

    # Ripristina i pesi migliori
    if early_stopping.best_model_weights is not None:
        model.load_state_dict(early_stopping.best_model_weights)
        print("Pesi migliori ripristinati.")

    # ==========================================
    # 4. MATRICE DI CONFUSIONE - VALIDAZIONE (miglior modello)
    # ==========================================
    model.eval()
    with torch.no_grad():
        val_probs = model(X_val)
        val_preds_final = (val_probs > 0.5).float()
        val_acc_final = accuracy_score(Y_val, val_preds_final)

    print(f"\nValidation Accuracy (miglior modello): {val_acc_final:.4f}")

    cm_val = confusion_matrix(Y_val.numpy(), val_preds_final.numpy())
    print("\nMatrice di Confusione - Validazione:")
    print(cm_val)

    fig_cm_val, ax_cm_val = plt.subplots(figsize=(6, 5))
    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=["Classe 0", "Classe 1"])
    disp_val.plot(ax=ax_cm_val, cmap='Blues', values_format='d')
    ax_cm_val.set_title(f'Matrice di Confusione - Validazione\n{architecture}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix_validation.png'), dpi=150)
    plt.close()

    # ==========================================
    # 5. TEST FINALE + MATRICE DI CONFUSIONE
    # ==========================================
    with torch.no_grad():
        test_probs = model(X_test)
        test_preds = (test_probs > 0.5).float()
        test_acc = accuracy_score(Y_test, test_preds)

    print(f"\nTest Accuracy: {test_acc:.4f}")

    cm_test = confusion_matrix(Y_test.numpy(), test_preds.numpy())
    print("\nMatrice di Confusione - Test:")
    print(cm_test)

    fig_cm_test, ax_cm_test = plt.subplots(figsize=(6, 5))
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["Classe 0", "Classe 1"])
    disp_test.plot(ax=ax_cm_test, cmap='Oranges', values_format='d')
    ax_cm_test.set_title(f'Matrice di Confusione - Test Finale\n{architecture}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix_test.png'), dpi=150)
    plt.close()

    # ==========================================
    # 6. GRAFICI: LOSS e ACCURACY
    # ==========================================
    epochs_range = range(1, len(loss_history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Grafico Loss ---
    axes[0].plot(epochs_range, loss_history, 'b-o', markersize=4, label='Train Loss')
    axes[0].plot(epochs_range, val_loss_history, 'r-o', markersize=4, label='Val Loss')
    axes[0].set_xlabel('Epoca')
    axes[0].set_ylabel('Loss (BCE)')
    axes[0].set_title(f'Train vs Val Loss ({architecture})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Grafico Accuracy ---
    axes[1].plot(epochs_range, train_acc_history, 'b-o', markersize=4, label='Train Accuracy')
    axes[1].plot(epochs_range, val_acc_history, 'r-o', markersize=4, label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Train vs Val Accuracy ({architecture})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150)
    plt.close()