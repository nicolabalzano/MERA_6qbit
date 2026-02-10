import time
import logging
import numpy as np
from preprocessing import data_load_and_process_mnist
from mera_circuit import construct_mera_vqc, construct_qnn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    start_global = time.time()
    logger.info("Starting MERA VQC Experiment")

    # --- Configuration ---
    NUM_QUBITS = 6
    CLASSES = 2 # Binary classification (e.g. 0 vs 1)
    # Using small subset for quick demonstration/testing. Increase for real results.
    NUM_EXAMPLES = 200 
    EPOCHS = 20  # For the classical AE if used (but it won't be used in tfgpu env without torch)
    MAXITER = NUM_EXAMPLES # Optimization iterations for VQC
    SEED = 42

    # --- 1. Data Loading ---
    logger.info("Loading and processing data...")
    # Note: Autoencoder is skipped if Torch is missing, default to PCA
    # We ask for PCA explicitly here to be safe and consistent
    X_train, X_test, Y_train, Y_test, prep_time = data_load_and_process_mnist(
        num_classes=CLASSES,
        all_samples=False,
        seed=SEED,
        num_examples_per_class=NUM_EXAMPLES,
        pca=True,
        n_features=NUM_QUBITS, # Feature map usually expects num_features = num_qubits
        epochs=EPOCHS 
    )
    logger.info(f"Data ready. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # --- 2. Quantum Circuit Construction ---
    logger.info("Constructing MERA 6-qubit circuit...")
    feature_map, ansatz = construct_mera_vqc(num_qubits=NUM_QUBITS)
    
    logger.info(f"Ansatz parameters: {ansatz.num_parameters}")
    logger.info(f"Ansatz depth: {ansatz.depth()}")

    # --- 3. VQC Model Setup ---
    logger.info("Initializing VQC...")
    vqc = construct_qnn(
        feature_map=feature_map,
        ansatz=ansatz,
        maxiter=MAXITER
    )

    # --- 4. Training ---
    logger.info("Starting VQC training...")
    try:        
        train_start = time.time()
        vqc.fit(X_train, Y_train)
        train_time = time.time() - train_start
        logger.info(f"Training completed in {train_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 5. Evaluation ---
    logger.info("Evaluating on Test Set...")
    try:
        score = vqc.score(X_test, Y_test)
        logger.info(f"Test Accuracy: {score:.4f}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

    total_time = time.time() - start_global
    logger.info(f"Experiment finished. Total time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
