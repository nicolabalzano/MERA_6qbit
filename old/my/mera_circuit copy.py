from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import StatevectorSampler

options = {'seed': 12345, 'shots': 4096}

def encoding_features_h_ry(num_qubits):
    qc = QuantumCircuit(num_qubits)
    feature_params = ParameterVector('x', num_qubits)

    for i in range(num_qubits):
        qc.h(i)
        qc.ry(feature_params[i], i)

    return qc

def MERA_block(num_qubits, prefix, **kwargs):
    """
    Creates a 2-qubit unitary block (RealAmplitudes by default)
    representing the "blue blocks" in MERA.
    """
    # Assuming the blue blocks operate on 2 qubits using RealAmplitudes strategy
    return RealAmplitudes(num_qubits=num_qubits, parameter_prefix=prefix, reps=1, **kwargs)

def MERA_6qubits(num_qubits=6, **kwargs):
    """
    Constructs a MERA ansatz for 6 qubits.
    Extends the concept of 4-qubit MERA to 6 qubits.
    
    Structure usually involves:
    - Layer 1: Disentanglers (Unitaries across boundaries or pairs)
    - Layer 2: Isometries (Coarse-graining)
    """
    qc = QuantumCircuit(num_qubits)
    
    # Layer 1: Disentanglers / Entanglers on adjacent pairs.
    # For a ring or linear chain, we place blocks.
    # If 6 qubits: 0-1, 1-2, 2-3, 3-4, 4-5, 5-0? 
    # MERA typically alternates and use linear chain. 
    # Let's assume a standard binary MERA pattern adaptation.
    
    # Layer 1 (Disentanglers): Acting on (even, odd) or overlapping pairs to reduce entanglement
    # Let's place 2-qubit blocks on (1,2), (3,4), (5,0) - wrapping around for ring, or (0,1), (2,3) etc if linear?
    # User mentioned "blocks in blue work only on 2 qubits".
    
    # Let's define a structure for 6 qubits based on general MERA principles:
    # 6 qubits -> Apply unitaries -> Coarse grain to 3? -> Unitaries -> Coarse grain to 1/2?
    
    # Disentanglers (First layer of filtering)
    # Between (0,1), (2,3), (4,5) are our "sites", disentanglers might sit BETWEEN them?
    # e.g., (5,0), (1,2), (3,4)
    
    # Block 1: Disentanglers
    qc.compose(MERA_block(2, 'dis_0', **kwargs), [5, 0], inplace=True)
    qc.compose(MERA_block(2, 'dis_1', **kwargs), [1, 2], inplace=True)
    qc.compose(MERA_block(2, 'dis_2', **kwargs), [3, 4], inplace=True)
    
    qc.barrier()
    
    # Layer 2: Isometries / Coarse graining blocks
    # Transforming pairs (0,1) -> 1 qubit (logical), (2,3) -> 1, (4,5) -> 1?
    # RealAmplitudes preserves qubit count, so we just entangle them.
    # Block on (0,1), (2,3), (4,5)
    qc.compose(MERA_block(2, 'iso_0', **kwargs), [0, 1], inplace=True)
    qc.compose(MERA_block(2, 'iso_1', **kwargs), [2, 3], inplace=True)
    qc.compose(MERA_block(2, 'iso_2', **kwargs), [4, 5], inplace=True)
    
    qc.barrier()
    
    # After this "Isometry" layer, we ideally focused information into specific qubits (e.g., 0, 2, 4 maybe?)
    # or we treat it as a deep circuit.
    # Next layer of MERA on the "surviving" effective qubits.
    # If we treat 1, 3, 5 as "discarded" or "ancilla" and 0, 2, 4 as next layer...
    # Let's entangle 0, 2, 4. 
    # Pair (0, 2)? Then (?, 4)?
    # Or common top block for 3 qubits? 
    # Since we use 2-qubit blocks, let's do (0, 2) and leave 4? Or ring (0,2), (2,4), (4,0)?
    
    # Let's add top-level entanglement for the remaining degrees of freedom
    qc.compose(MERA_block(2, 'top_0', **kwargs), [0, 2], inplace=True)
    qc.compose(MERA_block(2, 'top_1', **kwargs), [2, 4], inplace=True)
    
    # closing the loop if ring - BASED ON PAPER this is not implemented, but in the paper is only with 2 qubits
    #qc.compose(MERA_block(2, 'top_2', **kwargs), [4, 0], inplace=True)
   
    return qc

def construct_mera_vqc(num_qubits=6, measured_qubits=1):
    # Feature map (using the one from data_pipeline/ansatz)
    feature_map = encoding_features_h_ry(num_qubits)
    
    # Ansatz
    ansatz = QuantumCircuit(num_qubits)
    mera = MERA_6qubits(num_qubits)
    ansatz.compose(mera, inplace=True)
    
    # Measurement (usually we measure one or a few qubits)
    # If mapping to classes, we might measure parity or specific qubits.
    # existing ansatzy.py used specific measurement.
    
    # For VQC class we usually just pass the ansatz.
    # If we need manual measure, it's different. VQC class handles measurement mapping.
    
    return feature_map, ansatz

def construct_qnn(feature_map, ansatz, callback_graph=None, maxiter=100, interpret=None, output_shape=2) -> VQC:
    sampler = StatevectorSampler(seed=12345)
    
    # uniform initialization of parameter
    initial_point = [0.1] * ansatz.num_parameters # Small random or fixed
    
    classifier = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=maxiter),
        callback=callback_graph,
        interpret=interpret,
        loss='cross_entropy',
        output_shape=output_shape,
        initial_point=initial_point,
    )
    
    return classifier
