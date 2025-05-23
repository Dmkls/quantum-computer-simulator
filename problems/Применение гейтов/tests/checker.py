import math
import numpy as np
from quantum_circuit import QuantumCircuit
from reference_qc import ReferenceQuantumCircuit

def run_test(n, gate, targets, gate_name="_"):
    qc_ref = ReferenceQuantumCircuit(n)
    qc_student = QuantumCircuit(n)

    qc_ref.apply_gate(gate, targets)
    qc_student.apply_gate(gate, targets)

    if np.allclose(qc_ref.state, qc_student.state, atol=1e-8):
        print(f"Прошёл тест гейта {gate_name} для кубитов {targets}")
    else:
        print(f"Не прошёл тест гейта {gate_name} для кубитов {targets}")
        print("Ожидалось :", qc_ref.state)
        print("Получено  :", qc_student.state)

H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
P = lambda theta: np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)

if __name__ == "__main__":
    run_test(3, H, [0], gate_name="H")
    run_test(2, X, [1], gate_name="X")
    run_test(5, H, [0, 1], gate_name="H")
    run_test(5, X, [0, 1], gate_name="X")
    run_test(9, H, [0, 1, 2], gate_name="H")
    run_test(9, X, [0, 1, 2], gate_name="X")
    run_test(9, P(math.pi), [0, 1, 2], gate_name="P(pi)")
    run_test(9, P(-math.pi/2), [0, 1, 2, 3], gate_name="P(-pi/2)")
