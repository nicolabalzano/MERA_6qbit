import numpy as np
import tensorflow as tf
import torch
import pandas as pd
from torch.optim import Adam
from torch.nn import BCELoss
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mera_circuit_qiskit import create_qnn, EarlyStopping, QuantumMERAClassifier
from preprocessing import data_load_and_process_mnist
import os


def main():
    print("---Training Triplet Autoencoder and extracting features---")
    dataset = 'new' # 'mnist' or 'new'

    if dataset == 'mnist':
        X_train_np, X_test_np, Y_train_np, Y_test_np, _ = data_load_and_process_mnist(
            num_classes=2,
            all_samples=False,
            seed=42,
            num_examples_per_class=250,
            pca=False,
            n_features=6,
            epochs=50,
            margin=0.2,
            alpha=1.0,
            type_model='linear'
        )
        # Split train into train & val as done before
        X_val_np = X_train_np[-200:]
        Y_val_np = Y_train_np[-200:]
        X_train_np = X_train_np[:-200]
        Y_train_np = Y_train_np[:-200]
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        Y_train = torch.tensor(Y_train_np, dtype=torch.float32).reshape(-1, 1)
        
        X_val = torch.tensor(X_val_np, dtype=torch.float32)
        Y_val = torch.tensor(Y_val_np, dtype=torch.float32).reshape(-1, 1)
        
        X_test_np = X_test_np[:200]
        Y_test_np = Y_test_np[:200]
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        Y_test = torch.tensor(Y_test_np, dtype=torch.float32).reshape(-1, 1)

    if dataset == 'new':
        df_train_full = pd.read_csv('data/new_data/train.csv')
        df_test = pd.read_csv('data/new_data/test.csv')
        
        # 1. Seleziona 200 campioni per classe per la VALIDAZIONE dal TRAIN
        val_indices = []
        for class_label in [0, 1]:
            # Cerchiamo gli indici nel dataframe di train
            indices = df_train_full.index[df_train_full['classes'] == class_label].tolist()
            val_indices.extend(indices[:200])
        
        df_val = df_train_full.loc[val_indices]
        df_train_remaining = df_train_full.drop(val_indices)
    
        # Train (rimanente)
        X_train_np = df_train_remaining.drop(columns=['classes']).values
        Y_train_np = df_train_remaining['classes'].values
        
        # Validation (estratto dal train)
        X_val_np = df_val.drop(columns=['classes']).values
        Y_val_np = df_val['classes'].values
        
        # Test (integro)
        X_test_np = df_test.drop(columns=['classes']).values
        Y_test_np = df_test['classes'].values

        #Conversione in PyTorch Tensors
        
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        Y_train = torch.tensor(Y_train_np, dtype=torch.float32).reshape(-1, 1)
        
        X_val = torch.tensor(X_val_np, dtype=torch.float32)
        Y_val = torch.tensor(Y_val_np, dtype=torch.float32).reshape(-1, 1)
        
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        Y_test = torch.tensor(Y_test_np, dtype=torch.float32).reshape(-1, 1)
    
    
    print("\nData loaded. Tensor shapes:")
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_val:", X_val.shape, "Y_val:", Y_val.shape)
    print("X_test:", X_test.shape, "Y_test:", Y_test.shape)
    #'mera', 'RyCNOTRy', '(RyRy)CNOT(RyRx)', 'RxRzCNOT'
    architectures = ['mera', 'RyCNOTRy', '(RyRy)CNOT(RyRx)', 'RxRzCNOT']
    results = []

    for architecture in architectures:
        print(f"\n===== VQC Architecture: {architecture} =====")
        
        # Create directory for results
        out_dir = f"results_triple_enc_{architecture}"
        os.makedirs(out_dir, exist_ok=True)
        qnn, qc = create_qnn(architecture)
        qc.draw(output='mpl', filename=os.path.join(out_dir, f'mera_circuit_{architecture}.png'))

        model = QuantumMERAClassifier(qnn)
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = BCELoss()

        epochs = 50
        batch_size = 20
        dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        early_stopping = EarlyStopping(patience=5, min_delta=0.001)

        loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            loss_history.append(avg_loss)
            
            model.eval()
            with torch.no_grad():
                val_preds_prob = model(X_val)
                val_loss = criterion(val_preds_prob, Y_val).item()
                val_loss_history.append(val_loss)
                
                train_preds_prob = model(X_train)
                train_preds = (train_preds_prob > 0.5).float()
                train_acc = accuracy_score(Y_train, train_preds)
                train_acc_history.append(train_acc)
                
                val_preds = (val_preds_prob > 0.5).float()
                val_acc = accuracy_score(Y_val, val_preds)
                val_acc_history.append(val_acc)
            model.train()
                
            print(f"Epoch {epoch+1:2d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"\nEarly stopping attivato all'epoca {epoch+1}!")
                break

        if early_stopping.best_model_weights is not None:
            model.load_state_dict(early_stopping.best_model_weights)
            print("Pesi migliori ripristinati.")

        print("---TEST FINALE---")
        model.eval()
        with torch.no_grad():
            test_probs = model(X_test)
            test_preds = (test_probs > 0.5).float()
            test_acc = accuracy_score(Y_test, test_preds)
            test_f1 = f1_score(Y_test, test_preds)
            test_loss = criterion(test_probs, Y_test).item()
            
            val_probs = model(X_val)
            val_preds = (val_probs > 0.5).float()
            val_f1 = f1_score(Y_val, val_preds)
            val_loss_final = criterion(val_probs, Y_val).item()

        print(f"\nTest Accuracy (miglior modello): {test_acc:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # SALVATAGGIO GRAFICI LOSS e ACCURACY
        epochs_range = range(1, len(loss_history) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs_range, loss_history, 'b-o', markersize=4, label='Train Loss')
        axes[0].plot(epochs_range, val_loss_history, 'r-o', markersize=4, label='Val Loss')
        axes[0].set_xlabel('Epoca')
        axes[0].set_ylabel('Loss (BCE)')
        axes[0].set_title(f'Train vs Validation Loss ({architecture})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_range, train_acc_history, 'b-o', markersize=4, label='Train Accuracy')
        axes[1].plot(epochs_range, val_acc_history, 'r-o', markersize=4, label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'Train vs Validation Accuracy ({architecture})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150)
        plt.close()
        
        # MATRICE DI CONFUSIONE - VALIDAZIONE
        cm_val = confusion_matrix(Y_val.numpy(), val_preds.numpy())
        fig_cm_val, ax_cm_val = plt.subplots(figsize=(6, 5))
        disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=["Classe 0", "Classe 1"])
        disp_val.plot(ax=ax_cm_val, cmap='Blues', values_format='d')
        ax_cm_val.set_title(f'Matrice di Confusione - Validazione ({architecture})\nF1: {val_f1:.4f} | Loss: {val_loss_final:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'confusion_matrix_validation.png'), dpi=150)
        plt.close()

        # MATRICE DI CONFUSIONE - TEST FINALE
        cm_test = confusion_matrix(Y_test.numpy(), test_preds.numpy())
        fig_cm_test, ax_cm_test = plt.subplots(figsize=(6, 5))
        disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["Classe 0", "Classe 1"])
        disp_test.plot(ax=ax_cm_test, cmap='Oranges', values_format='d')
        ax_cm_test.set_title(f'Matrice di Confusione - Test Finale ({architecture})\nF1: {test_f1:.4f} | Loss: {test_loss:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'confusion_matrix_test.png'), dpi=150)
        plt.close()

        # Save result to list
        results.append({
            'Architecture': architecture,
            'Val Accuracy': val_acc.item() if hasattr(val_acc, 'item') else val_acc,
            'Val F1': val_f1.item() if hasattr(val_f1, 'item') else val_f1,
            'Val Loss': val_loss_final,
            'Test Accuracy': test_acc.item() if hasattr(test_acc, 'item') else test_acc,
            'Test F1': test_f1.item() if hasattr(test_f1, 'item') else test_f1,
            'Test Loss': test_loss
        })

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = 'risultati_architetture_triple_enc.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n==========\nTraining Complete.\nResults saved to {csv_path}\n==========")

if __name__ == "__main__":
    main()
