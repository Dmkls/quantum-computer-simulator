import numpy as np


class ReferenceQuantumCircuit:
    def __init__(self, num_qubits: int):
        """Создаёт вектор квантовой системы и задаёт ей состояние |0...0>

        :param num_qubits: количество кубитов во всей системе
        """
        self.num_qubits = num_qubits
        self.state = np.zeros((2 ** num_qubits,), dtype=complex)
        self.state[0] = 1

    def apply_gate(self, gate: np.ndarray, targets: list[int]):
        """Эталонное применение гейта

        :param gate: унитарная матрица 2x2, действующая как гейт
        :param targets: массив, содержащий индексы кубитов, на которые нужно подействовать гейтом
        :return: None
        """
        full_gate = 1
        for i in range(self.num_qubits):
            if i in targets:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, np.eye(2))
        self.state = full_gate @ self.state
