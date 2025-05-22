import numpy as np
import math
import fractions
from collections import Counter

class QuantumCircuit:
    def __init__(self, num_qubits):
        """Создаёт вектор квантовой системы и задаёт ей состояние |0...0>

        :param num_qubits: количество кубитов во всей системе
        """
        self.num_qubits = num_qubits
        self.state = np.zeros((2 ** num_qubits,), dtype=complex)
        self.state[0] = 1

    def apply_gate(self, gate, targets):
        full_gate = 1
        for i in range(self.num_qubits):
            if i in targets:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, np.eye(2))
        self.state = full_gate @ self.state

    def apply_controlled_gate(self, gate, control, target):
        size = 2 ** self.num_qubits
        result = np.zeros((size, size), dtype=complex)
        for i in range(size):
            binary = list(format(i, f'0{self.num_qubits}b'))
            if binary[self.num_qubits - 1 - control] == '1':
                j = i ^ (1 << (self.num_qubits - 1 - target))
                temp = gate[int(binary[self.num_qubits - 1 - target])][:]
                result[i, i] = temp[0]
                result[i, j] = temp[1]
            else:
                result[i, i] = 1
        self.state = result @ self.state

    def apply_controlled_phase(self, theta, control, target):
        size = 2 ** self.num_qubits
        matrix = np.eye(size, dtype=complex)
        for i in range(size):
            bin_state = format(i, f'0{self.num_qubits}b')
            if bin_state[self.num_qubits - 1 - control] == '1' and bin_state[self.num_qubits - 1 - target] == '1':
                matrix[i, i] *= np.exp(1j * theta)
        self.state = matrix @ self.state

    def measure(self, shots=1024):
        probs = np.abs(self.state) ** 2
        outcomes = [format(i, f'0{self.num_qubits}b') for i in range(2 ** self.num_qubits)]
        measurements = np.random.choice(outcomes, size=shots, p=probs)
        return dict(Counter(measurements))

# Гейты
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
P = lambda theta: np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)

def qft(circuit, qubits):
    """Делает прямое квантовые преобразование Фурье

    :param circuit: Квантовая схема, в которой выполняется преобразование
    :type circuit: QuantumCircuit
    :param qubits: список квбитов, к которым применяется обратное QFT
    :return: None
    """
    n = len(qubits)
    for i in range(n):
        circuit.apply_gate(H, [qubits[i]])
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            circuit.apply_controlled_phase(angle, qubits[j], qubits[i])

def inverse_qft(circuit, qubits):
    """Делает обратное квантовые преобразование Фурье

    :param circuit: Квантовая схема, в которой выполняется преобразование
    :type circuit: QuantumCircuit
    :param qubits: список квбитов, к которым применяется обратное QFT
    :return: None
    """
    n = len(qubits)
    for i in reversed(range(n)):
        for j in reversed(range(i + 1, n)):
            angle = -np.pi / (2 ** (j - i))
            circuit.apply_controlled_phase(angle, qubits[j], qubits[i])
        circuit.apply_gate(H, [qubits[i]])

def get_angles(a, n):
    """Используется для симуляции фазовых сдвигов в функции phiADD, строит массив углов фазовых сдвигов

    :param a: Число для прибавления к квантовому регистру
    :param n: Количество кубитов в верхнем регистре
    :return: Массив из n углов, соответствующих фазовым поворотам, которые реализуют добавление числа a к квантовому регистру
    :rtype: dict[float]
    """
    s = bin(a)[2:].zfill(n)
    angles = np.zeros(n)
    for i in range(n):
        for j in range(i, n):
            if s[j] == '1':
                angles[n - i - 1] += 1 / 2 ** (j - i)
        angles[n - i - 1] *= np.pi
    return angles

def ccphase(circuit, theta, ctrl1, ctrl2, target):
    """Выполняет дважды контролируемый фазовый сдвиг

    :param circuit: Квантовая схема, в которой выполняется операция
    :type circuit: QuantumCircuit
    :param theta: Угол, на котороый происходит сдвиг
    :param ctrl1: Индекс первого управляющего кубита
    :param ctrl2: Индекс второго управляющего кубита
    :param target: Целевой кубит
    :return: None
    """
    size = 2 ** circuit.num_qubits
    matrix = np.eye(size, dtype=complex)
    for i in range(size):
        bin_state = format(i, f'0{circuit.num_qubits}b')
        if bin_state[circuit.num_qubits - 1 - ctrl1] == '1' and bin_state[circuit.num_qubits - 1 - ctrl2] == '1' and bin_state[circuit.num_qubits - 1 - target] == '1':
            matrix[i, i] *= np.exp(1j * theta)
    circuit.state = matrix @ circuit.state

def phiADD(circuit, qubits, a, n, inv=False):
    """Выполняет прибавление числа a к регистру qubits в фазовом представлении
    с помощью поразрядных фазовых поворотов

    Применяет гейты вида P(θ) к каждому кубиту регистра, имитируя сложение a в базисе
    квантового преобразования Фурье
    При `inv=True` выполняется вычитание (обратный сдвиг фазы)

    :param circuit: Квантовая схема, в которой выполняется операция
    :type circuit: QuantumCircuit
    :param qubits: Регистр, к которому прибавляется (или из которого вычитается) число a
    :param a: Целое число, которое прибавляется по модулю
    :param n: Количество кубитов в регистре qubits
    :param inv: Флаг, указывающий на выполнение вычитания вместо прибавления
    :return: None
    """
    angles = get_angles(a, n)
    for i in range(n):
        angle = -angles[i] if inv else angles[i]
        circuit.apply_gate(P(angle), [qubits[i]])

def cphiADD(circuit, qubits, ctrl, a, n, inv=False):
    """Выполняет контролируемое прибавление числа a к квантовому регистру в фазовом представлении

    Реализует поразрядное прибавление фазовых сдвигов к каждому кубиту регистра,
    только если управляющий кубит `ctrl` находятся в состоянии |1⟩.
    При `inv=True` происходит вычитание (сдвиги с отрицательным углом)

    Операция применяется в базисе квантового преобразования Фурье и используется
    для построения модульной арифметики в алгоритме Шора

    :param circuit: Квантовая схема, в которой выполняется операция
    :type circuit: QuantumCircuit
    :param qubits: Регистр, к которому прибавляется (или из которого вычитается) число a
    :param ctrl: Индекс управляющего кубита
    :param a: Число, которое прибавляется (или вычитается)
    :param n: Количество кубитов в регистре qubits
    :param inv: Флаг, указывающий, нужно ли вычитать вместо прибавления (по умолчанию False)
    :return: None
    """
    angles = get_angles(a, n)
    for i in range(n):
        angle = -angles[i] if inv else angles[i]
        circuit.apply_controlled_phase(angle, ctrl, qubits[i])

def ccphiADD(circuit, qubits, ctrl1, ctrl2, a, n, inv=False):
    """Выполняет дважды контролируемое прибавление числа a к регистру qubits в фазовом представлении

    Реализует поразрядное прибавление фазовых сдвигов к каждому кубиту регистра,
    только если оба управляющих кубита находятся в состоянии |1⟩
    При `inv=True` происходит вычитание (сдвиги с отрицательным углом)

    :param circuit: Квантовая схема, в которой выполняется операция
    :type circuit: QuantumCircuit
    :param qubits: Регистр, к которому прибавляется (или из которого вычитается) число a
    :param ctrl1: Индекс первого управляющего кубита
    :param ctrl2: Индекс второго управляющего кубита
    :param a: Число, которое прибавляется (или вычитается)
    :param n: Количество кубитов в регистре qubits
    :param inv: Флаг, указывающий, нужно ли вычитать вместо прибавления (по умолчанию False)
    :return: None
    """
    angles = get_angles(a, n)
    for i in range(n):
        angle = -angles[i] if inv else angles[i]
        ccphase(circuit, angle, ctrl1, ctrl2, qubits[i])

def ccphiADDmodN(circuit, q, ctrl1, ctrl2, aux_bit, a, N, n):
    """Выполняет дважды контролируемое прибавление числа a по модулю N к регистру q

    Реализует следующую логику:
    - условно прибавляет a при ctrl1 = ctrl2 = 1,
    - вычитает N безусловно,
    - проверяет, произошло ли переполнение (если результат < 0),
    - в случае переполнения добавляет N обратно, чтобы результат был корректен по модулю N,
    - использует вспомогательный кубит aux_bit для хранения и последующего сброса флага переполнения


    :param circuit: Квантовая схема
    :type circuit: QuantumCircuit
    :param q: Регистр, к которому прибавляется число
    :param ctrl1: Индекс первого управляющего кубита
    :param ctrl2: Индекс второго управляющего кубита
    :param aux_bit: Вспомогательный кубит для отслеживания переполнения
    :param a: Число, которое прибавляется по модулю N
    :param N: Модуль
    :param n: Количество кубитов в регистре q
    :return: None
    """
    ccphiADD(circuit, q, ctrl1, ctrl2, a, n, inv=False) # условное прибавление a
    phiADD(circuit, q, N, n, inv=True)                  # безусловное вычитание N (получаем x + a - N)
    inverse_qft(circuit, q)                             # проверяем переполнение
    circuit.apply_gate(X, [q[-1]])
    circuit.apply_controlled_gate(X, q[-1], aux_bit)
    circuit.apply_gate(X, [q[-1]])
    qft(circuit, q)
    cphiADD(circuit, q, aux_bit, N, n, inv=False)       # если было переполнение — прибавляем N обратно
    ccphiADD(circuit, q, ctrl1, ctrl2, a, n, inv=True)  # Вычитаем a обратно, чтобы восстановить начальное значение x, если переполнение произошло
    inverse_qft(circuit, q)                             # вновь проверяем и сбрасываем aux_bit
    circuit.apply_gate(X, [q[-1]])
    circuit.apply_controlled_gate(X, q[-1], aux_bit)
    circuit.apply_gate(X, [q[-1]])
    qft(circuit, q)
    ccphiADD(circuit, q, ctrl1, ctrl2, a, n, inv=False) # прибавляем a, так как не будет переполнения

def egcd(a, b):
    """Реализует расширенный алгоритм Евклида

    Находит наибольший общий делитель g = gcd(a, b), а также такие целые коэффициенты x и y, что:

        a * x + b * y = g

    :param a: Первое целое число
    :param b: Второе целое число
    :return: Кортеж (g, x, y), где g = gcd(a, b) и a * x + b * y = g
    :rtype: tuple[int, int, int]
    """
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = egcd(b % a, a)
        return g, x - (b // a) * y, y

def modinv(a, m):
    """Вычисляет обратный элемент по модулю m, то есть такое x, что a·x ≡ 1 mod m

    Использует расширенный алгоритм Евклида. Если числа не взаимно просты, вызывается исключение

    :param a: Число, для которого ищется обратный элемент
    :param m: Модуль, по которому вычисляется обратное
    :return: Обратное число x по модулю m, такое что (a * x) % m == 1
    :raises Exception: Если обратного элемента не существует (a и m не взаимно просты)
    """
    g, x, _ = egcd(a, m)
    if g != 1:
        raise Exception('Обратного по модулю не существует')
    return x % m

def cMULTmodN(circuit, ctrl, x_qubits, out_qubits, aux_bit, a, N, n):
    """Выполняет контролируемое умножение регистра на число a по модулю N

    Реализует операцию |x⟩ ⊗ |y⟩ → |x⟩ ⊗ |(y · a^x) mod N⟩, если управляющий кубит установлен

    Операция выполняется поразрядно в фазовом представлении с использованием квантового преобразования Фурье

    :param circuit: Квантовая схема, в которой выполняется операция
    :type circuit: QuantumCircuit
    :param ctrl: Индекс управляющего кубита, активирующего операцию
    :param x_qubits: Регистр, содержащий число x (в показателе степени)
    :param out_qubits: Регистр, который будет умножен на a^x mod N
    :param aux_bit: Вспомогательный кубит для контроля переполнения при mod N
    :param a: Множитель, основание степени
    :param N: Модуль
    :param n: Количество кубитов в регистрах (разрядность)
    :return: None
    """
    qft(circuit, out_qubits)
    for i in range(n):
        factor = (pow(2, i) * a) % N
        ccphiADDmodN(circuit, out_qubits, x_qubits[i], ctrl, aux_bit, factor, N, n)
    inverse_qft(circuit, out_qubits)

def initialize_shor_circuit(N):
    """Подготавливает начальное квантовое состояние для алгоритма Шора:

    верхний регистр инициализируется в равномерную суперпозицию,
    нижний регистр — в |1⟩, вспомогательный кубит — в |0⟩

    :param N: Число, подлежащее факторизации
    :type N: int
    :return:
        - qc (QuantumCircuit): Инициализированная квантовая схема
        - up_reg (list[int]): Верхний регистр (для значений x)
        - down_reg (list[int]): Нижний регистр (для хранения a^x mod N)
        - aux_bit (int): Вспомогательный кубит для контроля арифметики mod N
        - n (int): Количество кубитов в регистрах (разрядность)
    :rtype: tuple[QuantumCircuit, list[int], list[int], int, int]
    """

    n = math.ceil(math.log2(N))
    num_qubits = 2 * n + 1
    qc = QuantumCircuit(num_qubits)
    up_reg = list(range(0, n))
    down_reg = list(range(n, 2 * n))
    aux_bit = 2 * n
    qc.apply_gate(H, up_reg)
    qc.apply_gate(X, [down_reg[0]])

    return qc, up_reg, down_reg, aux_bit, n

def apply_controlled_exponentiation(qc, up_reg, down_reg, aux_bit, a, N, n):
    """Выполняет последовательное контролируемое модульное возведение в степень:

    реализует унитарное преобразование |x⟩ ⊗ |1⟩ → |x⟩ ⊗ |a^x mod N⟩.

    Для каждого бита xᵢ верхнего регистра управляемо умножает нижний регистр на a^{2^i} mod N

    :param qc: Квантовая схема, в которой происходит вычисление
    :type qc: QuantumCircuit
    :param up_reg: Верхний регистр, содержащий биты значения x
    :param down_reg: Нижний регистр, в который записывается результат a^x mod N
    :param aux_bit: Вспомогательный кубит для контроля переполнений в модульной арифметике
    :param a: Основание степени
    :param N: Модуль
    :param n: Количество кубитов в регистрах
    :type n: int
    :return: None
    """
    for i in range(len(up_reg)):
        exponent = pow(a, 2 ** i, N)
        cMULTmodN(qc, up_reg[i], down_reg, down_reg, aux_bit, exponent, N, n)

def find_period(x, n, N, a):
    """Ищет период функции f(x) = a^x mod N на основе результата измерения квантового регистра

    :param x: Результат измерения верхнего регистра (целое число)
    :param n: Количество кубитов в верхнем регистре (разрядность измерения)
    :param N: Число, подлежащее факторизации
    :param a: Основание степени, взаимно простое с N
    :return: Найденный период r, если удалось, иначе None
    :rtype: int | None
    """

    if x == 0:
        return None
    T = 2 ** n
    frac = fractions.Fraction(x, T).limit_denominator(N)
    r = frac.denominator
    if pow(a, r, N) == 1:
        return r
    return None

def measure_and_analyze(qc, up_reg, shots=1024):
    """Измеряет верхний регистр схемы после применения обратного квантового преобразования Фурье

    :param qc: Квантовая схема (экземпляр QuantumCircuit)
    :param up_reg: Список индексов кубитов верхнего регистра, подлежащих измерению
    :param shots: Количество измерений (по умолчанию 1024)
    :return: Словарь вида {'битовая строка': число повторений}
    :rtype: dict[str, int]
    """

    inverse_qft(qc, up_reg)
    result = qc.measure(shots=shots)
    # print("Результаты измерения:")
    # for k, v in result.items():
    #     print(f"{k} — {v} раз")
    return result

def shor(N, a, shots=1024):
    """Выполняет полный цикл алгоритма Шора с квантовой симуляцией и классической постобработкой

    :param N: Целое число, подлежащее факторизации
    :type N: int
    :param a: Целое число, взаимно простое с N (основание степени)
    :type a: int
    :param shots: Количество симулированных измерений (по умолчанию 1024)
    :type shots: int
    :return: None. Результаты выводятся в консоль
    """
    qc, up_reg, down_reg, aux_bit, n = initialize_shor_circuit(N)
    apply_controlled_exponentiation(qc, up_reg, down_reg, aux_bit, a, N, n)
    result = measure_and_analyze(qc, up_reg, shots)
    for bitstring in result:
        x = int(bitstring, 2)
        r = find_period(x, 2 * n, N, a)
        if r:
            print(f"Найден период r = {r}")
            factor1 = math.gcd(pow(a, r//2) - 1, N)
            factor2 = math.gcd(pow(a, r//2) + 1, N)
            if factor1 not in [1, N] and factor2 not in [1, N]:
                print(f"Найдено: {factor1} × {factor2} = {N}")
                return
    print("Не удалось найти период — попробуйте с другим 'a'")

if __name__ == "__main__":
    N = int(input("Введите N для факторизации: "))
    a = int(input(f"Введите a (взаимнопростое с {N}): "))
    shor(N, a, shots=512)
