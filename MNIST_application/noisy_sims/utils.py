'''
Utility file containing helper functions.
'''

import torch
import numpy as np
import random
import pennylane as qml
import numpy.random as npr
from qiskit_ibm_runtime.fake_provider import FakeBrisbane # Import target backend of your choice

# Define target backend and layout qubits/coupling map
backend = FakeBrisbane()
layout_qubits = [0,1,2,3,4,5,6,7]
layout_coupling = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]

# Define pennylane-to-qiskit mapping for basis gates
basis_gate_map = {qml.RZ: 'rz', qml.ECR: 'ecr', qml.SX: 'sx', qml.PauliX: 'x', qml.Identity: 'id'}

def set_seed(seed=123):
    # Fix the seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False
    # np.random.seed(seed)
    # random.seed(seed)

def obtain_backend_noise(layout_qubits=layout_qubits, layout_coupling=layout_coupling, backend=FakeBrisbane(),):
    '''
    Obtain backend noise for coupling of choice
    '''
    # Obtain backend properties
    properties = backend.properties()

    # Extract T1 and T2 times
    t1_times = {}
    t2_times = {}
    for qubit in layout_qubits:
        t1_times[qubit] = properties.t1(qubit)
        t2_times[qubit] = properties.t2(qubit)

    # Extract gate errors and times based on the layout
    gate_errors = {}
    gate_times = {}

    # Filter gates for the specified layout_qubits and layout_coupling
    for gate in properties.gates:
        gate_name = gate.gate
        qubits = gate.qubits
        qubits_tuple = tuple(qubits)

        # Only consider gates involving qubits from layout_qubits and couplings in layout_coupling
        if all(q in layout_qubits for q in qubits) and (len(qubits_tuple) == 1 or qubits_tuple in layout_coupling):
            for param in gate.parameters:
                if param.name == 'gate_error':
                    gate_errors[(gate_name, qubits_tuple)] = param.value
                if param.name == 'gate_length':
                    gate_times[(gate_name, qubits_tuple)] = param.value * 1e-9  # Convert to seconds

    # Extract readout errors
    readout_errors = {}
    for qubit in layout_qubits:
        readout_errors[qubit] = properties.readout_error(qubit)
    
    return t1_times, t2_times, gate_times, gate_errors, readout_errors

# Obtain target coupling characteristics
t1_times, t2_times, gate_times, gate_errors, readout_errors = obtain_backend_noise()

# def custom_rx(phi, wires):
#     gate_list =  [
#         qml.RZ(np.pi/2, wires=wires),
#         qml.SX(wires=wires),
#         qml.RZ(np.pi + phi, wires=wires),
#         qml.SX(wires=wires),
#         qml.RZ(5*np.pi/2, wires=wires)
#     ]
#     for gate in gate_list:
#         apply_gate_with_noise(type(gate), basis_gate_map[type(gate)], gate.wires, gate.parameters)

def custom_rx(phi, wires):
    gate_info = [
        (qml.RZ, [np.pi/2], [wires]),
        (qml.SX, [], [wires]),
        (qml.RZ, [np.pi + phi], [wires]),
        (qml.SX, [], [wires]),
        (qml.RZ, [5*np.pi/2], [wires])
    ]
    for gate_type, params, gate_wires in gate_info:
        apply_gate_with_noise(gate_type, basis_gate_map[gate_type], gate_wires, params)

# def custom_ry(phi, wires):
#     gate_list = [
#         qml.SX(wires=wires),
#         qml.RZ(np.pi + phi, wires=wires),
#         qml.SX(wires=wires),
#         qml.RZ(3*np.pi, wires=wires)
#     ]
#     for gate in gate_list:
#         apply_gate_with_noise(type(gate), basis_gate_map[type(gate)], gate.wires, gate.parameters)

def custom_ry(phi, wires):
    gate_info = [
        (qml.SX, [], [wires]),
        (qml.RZ, [np.pi + phi], [wires]),
        (qml.SX, [], [wires]),
        (qml.RZ, [3*np.pi], [wires])
    ]
    for gate_type, params, gate_wires in gate_info:
        apply_gate_with_noise(gate_type, basis_gate_map[gate_type], gate_wires, params)


def custom_rot(phi, theta, omega, wires):
    apply_gate_with_noise(qml.RZ, 'rz', [wires], [phi])
    custom_ry(theta, wires)
    apply_gate_with_noise(qml.RZ, 'rz', [wires], [omega])

# def custom_CNOT(wires):
#     gate_list = [
#         qml.RZ(-np.pi/2, wires=wires[0]),
#         qml.RZ(-np.pi, wires=wires[1]),
#         qml.SX(wires=wires[1]),
#         qml.RZ(-np.pi, wires=wires[1]),
#         qml.ECR(wires=wires),
#         qml.PauliX(wires=wires[0])
#     ]
#     for gate in gate_list:
#         apply_gate_with_noise(type(gate), basis_gate_map[type(gate)], gate.wires, gate.parameters)

def custom_CNOT(wires):
    gate_info = [
        (qml.RZ, [-np.pi/2], [wires[0]]),
        (qml.RZ, [-np.pi], [wires[1]]),
        (qml.SX, [], [wires[1]]),
        (qml.RZ, [-np.pi], [wires[1]]),
        (qml.ECR, [], wires),
        (qml.PauliX, [], [wires[0]])
    ]
    for gate_type, params, gate_wires in gate_info:
        apply_gate_with_noise(gate_type, basis_gate_map[gate_type], gate_wires, params)


def depolarizing_error(qubits, error_rate):
    """
    Simulate depolarizing error by applying a random Pauli gate to each qubit
    with a probability equal to the error_rate.
    """
    for q in qubits:
        if npr.rand() < error_rate:
            # Apply a random Pauli error
            pauli_errors = [qml.PauliX, qml.PauliY, qml.PauliZ]
            error_gate = npr.choice(pauli_errors)
            error_gate(wires=q)

def readout_error(qubits):
    """
    Simulate readout error by applying a bit-flip (PauliX gate) to each qubit
    with a probability equal to its readout error rate.
    """
    for q in qubits:
        error_rate = readout_errors.get(q, 0)
        if npr.rand() < error_rate:
            qml.PauliX(wires=q)

def thermal_relaxation_error(qubits, gate_time, t1_times, t2_times):
    """
    Simulate thermal relaxation error by applying PauliX (decay to |0>) and
    PauliZ (dephasing) gates based on calculated probabilities for each qubit.
    """
    physical_qubits = map_logical_to_physical(qubits, layout_qubits)
    for logical_q, physical_q in zip(qubits, physical_qubits):
        T1 = t1_times.get(physical_q, 209.83e-6)  # Default T1 if not found
        T2 = t2_times.get(physical_q, 144.81e-6)  # Default T2 if not found
        # Probabilities for decay and dephasing
        p_reset = 1 - np.exp(-gate_time / T1)
        p_dephase = 1 - np.exp(-gate_time / T2)
        # Simulate relaxation to |0> with probability p_reset
        if npr.rand() < p_reset:
            qml.PauliX(wires=logical_q)
        # Simulate dephasing with probability p_dephase
        if npr.rand() < p_dephase:
            qml.PauliZ(wires=logical_q)

def apply_gate_with_noise(gate_func, gate_name, qubits, params=[], layout_qubits=layout_qubits):
    '''
    Define a function to apply gates with noise
    '''
    # Map logical qubits to physical qubits
    physical_qubits = map_logical_to_physical(qubits, layout_qubits)
    # Apply the gate
    gate_func(*params, wires=qubits)
    # Get gate key
    gate_key = (gate_name, tuple(physical_qubits))
    # Default values if gate properties are not found
    default_gate_time = 660e-9  # 660 ns
    default_error_rate = 7.821e-3  # Adjust as needed
    # Get gate time and error rate
    gate_time = gate_times.get(gate_key, default_gate_time)
    error_rate = gate_errors.get(gate_key, default_error_rate)
    # Apply depolarizing error after gate
    depolarizing_error(qubits, error_rate)
    # Apply thermal relaxation error after gate
    thermal_relaxation_error(qubits, gate_time, t1_times, t2_times)

def map_logical_to_physical(qubits, layout_qubits):
    """
    Map logical qubits (0, 1, 2) to physical qubits based on layout_qubits.
    """
    return [layout_qubits[q] for q in qubits]


def qubit_numbering_mapping(layout_qubits, layout_coupling):
    '''
    Map layout qubits and coupling from 0,1,2,...
    '''
    num_qubits = len(layout_qubits)
    qubit_numbering_dict = {}
    for cnt in range(num_qubits):
        qubit_numbering_dict[layout_qubits[cnt]]=cnt
    new_layout_qubits = list(range(num_qubits))
    new_layout_coupling = []
    for coupling in layout_coupling:
        new_layout_coupling.append([qubit_numbering_dict[coupling[0]], qubit_numbering_dict[coupling[1]]])
    return new_layout_qubits, new_layout_coupling