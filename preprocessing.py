import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from itertools import combinations
import torch.optim as optim
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import tensorflow as tf
from torchvision import datasets, transforms
import random
import time

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)



# MNIST autoencoder
class TripletAutoencoder(nn.Module):
    def __init__(self, input_dim=784, bottleneck_dim=8):
        super(TripletAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed

def generate_triplets(labels):
    triplets = []
    labels = labels.cpu().numpy()

    for label in set(labels):
        pos_idx = [i for i in range(len(labels)) if labels[i] == label]
        neg_idx = [i for i in range(len(labels)) if labels[i] != label]
        if len(pos_idx) < 2:
            continue
        for anchor, positive in combinations(pos_idx, 2):
            negative = random.choice(neg_idx)
            triplets.append((anchor, positive, negative))
    return triplets


def extract_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        encoded = model.encoder(data_tensor)
        return encoded.cpu().numpy()
    

def train_triplet_autoencoder(model, X, y, n_epochs=100, batch_size=32, lr=1e-3, margin=1.0, alpha=0.5):
    model.to(device)
    criterion_recon = nn.MSELoss()
    criterion_triplet = nn.TripletMarginLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            emb, xb_recon = model(xb)

            loss_recon = criterion_recon(xb_recon, xb)

            # Triplet loss
            triplets = generate_triplets(yb)
            if triplets:
                anchor = torch.stack([emb[a] for a, _, _ in triplets])
                positive = torch.stack([emb[p] for _, p, _ in triplets])
                negative = torch.stack([emb[n] for _, _, n in triplets])
                loss_triplet = criterion_triplet(anchor, positive, negative)
                loss = alpha * loss_recon + loss_triplet
            else:
                loss = loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss:.4f}")

    model.to('cpu')
    return model



    

# feature map
def encoding_features_h_ry(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.h(i)
        qc.ry(feature_params[i], i)

    return qc

def data_load_and_process_mnist(
        num_classes,
        all_samples,
        seed,
        num_examples_per_class,
        pca=True,
        n_features=8,
        epochs = 300,
        margin=.2,
        alpha=1.,
        type_model='linear'
):
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    x_train = mnist_train.data.numpy().astype(np.float32) / 255.0
    y_train = mnist_train.targets.numpy()

    x_test = mnist_test.data.numpy().astype(np.float32) / 255.0
    y_test = mnist_test.targets.numpy()

    if type_model != 'linear':
        x_train = np.expand_dims(x_train, 1)
        x_test = np.expand_dims(x_test, 1)
    else:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    if not all_samples:
        selected_indices = []

        for class_label in range(10):
            indices = np.where(y_train == class_label)[0][:num_examples_per_class]
            selected_indices.extend(indices)

        x_train_subset = x_train[selected_indices]
        y_train_subset = y_train[selected_indices]

        shuffle_indices = np.random.permutation(len(x_train_subset))
        x_train = x_train_subset[shuffle_indices]
        y_train = y_train_subset[shuffle_indices]

    logger.info("Shape of subset training data: {}", x_train.shape)
    logger.info("Shape of subset training labels: {}", y_train.shape)

    mask_train = np.isin(y_train, range(0, num_classes))
    mask_test = np.isin(y_test, range(0, num_classes))

    X_train = x_train[mask_train].reshape(-1, 784)
    X_test = x_test[mask_test].reshape(-1, 784)


    Y_train = y_train[mask_train]
    Y_test = y_test[mask_test]

    logger.info("Shape of subset training data: {}", X_train.shape)
    logger.info("Shape of subset training labels: {}", Y_train.shape)
    logger.info("Shape of testing data: {}", X_test.shape)
    logger.info("Shape of testing labels: {}", Y_test.shape)
    if pca:
        start = time.time()
        pca = PCA(n_features)
        X_train = pca.fit_transform(X_train)
        end = time.time()
        total_time = end - start
        X_test = pca.transform(X_test)
    else:
        autoencoder = TripletAutoencoder(bottleneck_dim=n_features)

        start = time.time()
        autoencoder = train_triplet_autoencoder(
            autoencoder,
            X_train,
            Y_train,
            n_epochs=epochs,
            batch_size=100,
            lr=1e-3,
            margin=margin,
            alpha=alpha,
        )
        end = time.time()
        total_time = end - start

        X_train = extract_embeddings(autoencoder, X_train)
        X_test = extract_embeddings(autoencoder, X_test)

    X_train = (X_train - X_train.min()) * (np.pi / (X_train.max() - X_train.min()))
    X_test = (X_test - X_test.min()) * (np.pi / (X_test.max() - X_test.min()))
    """
    # First, calculate scale parameters from training data ONLY to avoid leakage and ensure consistency
    x_min = X_train.min()
    x_max = X_train.max()
    
    # Scale both using training parameters
    X_train = (X_train - x_min) * (np.pi / (x_max - x_min))
    X_test = (X_test - x_min) * (np.pi / (x_max - x_min))
    """

    return X_train, X_test, Y_train, Y_test, total_time


def data_load_and_process_ctscan(
        dataset_path,
        num_examples_per_class,
        seed,
        img_size=28,
        pca=False,
        n_features=6,
        epochs=50,
        margin=0.2,
        alpha=1.0,
        test_split=0.2,
):
    """
    Carica le immagini CT-scan SARS-CoV-2 (COVID / non-COVID), le
    ridimensiona a img_size x img_size in scala di grigi, le appiattisce
    e applica la stessa pipeline TripletAutoencoder usata per MNIST.

    Restituisce: X_train, X_test, Y_train, Y_test, total_time
    """
    from PIL import Image
    import os

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    class_dirs = {0: os.path.join(dataset_path, 'non-COVID'),
                  1: os.path.join(dataset_path, 'COVID')}

    images_all = []
    labels_all = []

    for label, folder in class_dirs.items():
        files = np.array(sorted([f for f in os.listdir(folder)
                                  if f.lower().endswith('.png')]))
        np.random.shuffle(files)
        files = files[:num_examples_per_class]
        for fname in files:
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert('L')          # grayscale
            img = img.resize((img_size, img_size), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            images_all.append(arr.flatten())
            labels_all.append(label)

    X = np.array(images_all)
    Y = np.array(labels_all)

    # Shuffle everything
    perm = np.random.permutation(len(X))
    X, Y = X[perm], Y[perm]

    # Train / test split
    n_test = int(len(X) * test_split)
    X_test, Y_test = X[:n_test], Y[:n_test]
    X_train_full, Y_train_full = X[n_test:], Y[n_test:]

    # Separate the 200 samples used as Validation
    num_val = 200
    if len(X_train_full) > num_val:
        X_train_pure = X_train_full[:-num_val]
        Y_train_pure = Y_train_full[:-num_val]
        X_val = X_train_full[-num_val:]
        Y_val = Y_train_full[-num_val:]
    else:
        # Fallback di sicurezza nel caso i campioni siano troppo pochi
        X_train_pure = X_train_full
        Y_train_pure = Y_train_full
        X_val = np.array([])
        Y_val = np.array([])

    input_dim = img_size * img_size

    if pca:
        from sklearn.decomposition import PCA as _PCA
        start = time.time()
        pca_model = _PCA(n_features)
        X_train_pure = pca_model.fit_transform(X_train_pure)
        if len(X_val) > 0:
            X_val = pca_model.transform(X_val)
        X_test = pca_model.transform(X_test)
        end = time.time()
        total_time = end - start
    else:
        autoencoder = TripletAutoencoder(input_dim=input_dim, bottleneck_dim=n_features)
        start = time.time()
        # TRAIN DELL'ENCODER SOLO SUL TRAIN PURO (Senza i dati val)
        autoencoder = train_triplet_autoencoder(
            autoencoder,
            X_train_pure,
            Y_train_pure,
            n_epochs=epochs,
            batch_size=100,
            lr=1e-3,
            margin=margin,
            alpha=alpha,
        )
        end = time.time()
        total_time = end - start
        
        # Estraiamo per i 3 set
        X_train_pure = extract_embeddings(autoencoder, X_train_pure)
        if len(X_val) > 0:
            X_val = extract_embeddings(autoencoder, X_val)
        X_test = extract_embeddings(autoencoder, X_test)

    # Normalise to [0, pi] as done for MNIST
    X_train_pure = (X_train_pure - X_train_pure.min()) * (np.pi / (X_train_pure.max() - X_train_pure.min()))
    if len(X_val) > 0:
        X_val = (X_val - X_val.min()) * (np.pi / (X_val.max() - X_val.min()))
    X_test = (X_test - X_test.min()) * (np.pi / (X_test.max() - X_test.min()))

    return X_train_pure, X_val, X_test, Y_train_pure, Y_val, Y_test, total_time

