import numpy as np
from qps_ref import shor as ref_shor
from qps_sol import shor as sol_shor


def run_test(N, a):
    ref_result = ref_shor(N, a, 512)
    sol_result = sol_shor(N, a, 512)

    is_success = None

    if ref_result is None and sol_result is None:
        is_success = True
    elif isinstance(ref_result, list) and isinstance(sol_result, list):
        if np.allclose(ref_result, sol_result, atol=1e-8):
            is_success = True
    else:
        is_success = False

    if is_success:
        print(f"Прошёл тест алгоритма Шора на симмуляторе квантового компьютера для числа {N} и основания {a}")
    else:
        print(f"Прошёл тест алгоритма Шора для числа {N} и основания {a}")
        print("Ожидалось :", ref_shor)
        print("Получено  :", sol_shor)

if __name__ == "__main__":
    run_test(2, 1)
    run_test(6, 1)
    run_test(6, 5)
    run_test(8, 2)
    run_test(7, 5)
    run_test(15, 2)
    run_test(21, 2)

