import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ==========================================
# 1. PREPARAZIONE DATASET (MNIST)
# ==========================================
def load_and_process_mnist():
    print("Caricamento e preprocessing MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    #solo due classi per classificazione binaria
    mask_train = (y_train == 0) | (y_train == 1)
    mask_test = (y_test == 0) | (y_test == 1)

    x_train, y_train = x_train[mask_train], y_train[mask_train]
    x_test, y_test = x_test[mask_test], y_test[mask_test]

    # Riduciamo il dataset per velocizzare la demo (opzionale)
    x_train = x_train[:500]
    y_train = y_train[:500]
    x_test = x_test[:100]
    y_test = y_test[:100]

    # Resize delle immagini a 6 pixel (3x2) per adattarsi ai 6 qubit
    # Il paper usava un crop + downsampling su immagini 37x37 ottenendo 4 o 6 pixel.
    x_train_resized = tf.image.resize(x_train[..., np.newaxis], (3, 2)).numpy().reshape(-1, 6)
    x_test_resized = tf.image.resize(x_test[..., np.newaxis], (3, 2)).numpy().reshape(-1, 6)

    # Normalizzazione [0, pi] come da paper (Sezione III.A e Fig. 3)
    # MNIST è 0-255. Scaliamo a 0-1 poi moltiplichiamo per pi.
    x_train_norm = (x_train_resized / 255.0) * np.pi
    x_test_norm = (x_test_resized / 255.0) * np.pi

    # Convertiamo le label 0/1 in formato +/- 1 per coerenza con PauliZ (opzionale)
    # Oppure manteniamo 0/1 per CrossEntropy. Manteniamo 0/1.
    return x_train_norm, y_train, x_test_norm, y_test

X_train, Y_train, X_test, Y_test = load_and_process_mnist()
print(f"Dataset pronto. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ==========================================
# 2. DEFINIZIONE CIRCUITO
# ==========================================
n_qubits = 6
dev = qml.device("default.qubit", wires=n_qubits)

def feature_map(data):
    """Encoding Ry"""
    for i, val in enumerate(data):
        qml.RY(val, wires=i)

def unitary_block(weights, wires):
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

@qml.qnode(dev)
def circuit_node(data, weights):
    feature_map(data)
        
    # Layer 1
    unitary_block(weights[0], wires=[1, 2])
    unitary_block(weights[1], wires=[3, 4])
    
    # Layer 2
    unitary_block(weights[2], wires=[0, 1])
    unitary_block(weights[3], wires=[2, 3])
    unitary_block(weights[4], wires=[4, 5])
    
    # Layer 3
    unitary_block(weights[5], wires=[1, 4])
    
    # Layer 4
    unitary_block(weights[6], wires=[1, 2])
    unitary_block(weights[7], wires=[3, 4])
    
    # Layer 5
    unitary_block(weights[8], wires=[1, 4])
    
    return qml.expval(qml.PauliZ(4)) # misurazione

# ==========================================
# 3. FUNZIONI DI COSTO E TRAINING
# ==========================================

def variational_classifier(weights, bias, x):
    # Esegue il circuito
    exp_val = circuit_node(x, weights)
    # bias classico
    return exp_val + bias

def cost(weights, bias, X, Y):
    # Calcoliamo le predizioni per tutto il batch
    predictions_exp = np.array([variational_classifier(weights, bias, x) for x in X])
    
    # il paper usa Cross Entropy (Eq 9)
    # Prob(y=1) = (1 - expectation) / 2
    prob_1 = (1 - predictions_exp) / 2
    
    # Clipping per evitare log(0)
    prob_1 = np.clip(prob_1, 1e-15, 1 - 1e-15)
    
    # Binary Cross Entropy
    # L = - [ y * log(p) + (1-y) * log(1-p) ]
    loss = -np.mean(Y * np.log(prob_1) + (1 - Y) * np.log(1 - prob_1))
    
    return loss

def predict(weights, bias, X):
    # Predizione finale (Hard threshold)
    pred_vals = np.array([variational_classifier(weights, bias, x) for x in X])
    # Se exp < 0 -> stato |1> -> Classe 1
    # Se exp > 0 -> stato |0> -> Classe 0
    return np.where(pred_vals < 0, 1, 0)

# ==========================================
# 4. LOOP DI TRAINING
# ==========================================

# Inizializzazione pesi (9 blocchi x 2 parametri)
np.random.seed(42)
weights = np.random.randn(9, 2, requires_grad=True) * 0.1
bias = np.array(0.0, requires_grad=True)

# Iperparametri (Paper: lr=10^-2 per quantum, batch size 50-100)
opt = qml.AdamOptimizer(stepsize=0.01)
batch_size = 20
epochs = 15

print("\nInizio Training...")
loss_history = []

for epoch in range(epochs):
    # Shuffle del dataset
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    Y_train_shuffled = Y_train[indices]
    
    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        batch_x = X_train_shuffled[i : i + batch_size]
        batch_y = Y_train_shuffled[i : i + batch_size]
        
        # Update passo ottimizzatore
        weights, bias = opt.step(cost, weights, bias, X=batch_x, Y=batch_y)
    
    # Calcolo loss e accuracy su tutto il train set per monitoraggio
    current_loss = cost(weights, bias, X_train, Y_train)
    predictions = predict(weights, bias, X_train)
    acc = accuracy_score(Y_train, predictions)
    loss_history.append(current_loss)
    
    print(f"Epoch {epoch+1:2d} | Loss: {current_loss:.4f} | Train Acc: {acc:.4f}")

# ==========================================
# 5. TEST FINALE
# ==========================================
print("\nValutazione sul Test Set...")
test_predictions = predict(weights, bias, X_test)
test_acc = accuracy_score(Y_test, test_predictions)

print(f"Test Accuracy: {test_acc:.4f}")