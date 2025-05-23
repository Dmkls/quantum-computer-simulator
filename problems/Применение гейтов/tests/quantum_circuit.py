import numpy as np

class QuantumCircuit:
    def __init__(self, num_qubits: int):
        """Создаёт вектор квантовой системы и задаёт ей состояние |0...0>

        :param num_qubits: количество кубитов во всей системе
        """
        self.num_qubits = num_qubits
        self.state = np.zeros((2 ** num_qubits,), dtype=complex)
        self.state[0] = 1

    def apply_gate(self, gate: np.ndarray, targets: list[int]):
        """Действует гейтом на заданные кубиты

        :param gate: унитарная матрица 2x2, действующая как гейт
        :param targets: массив, содержащий индексы кубитов, на которые нужно подействовать гейтом
        :return: None
        """
        raise NotImplementedError("Реализуйте этот метод")