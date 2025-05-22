def shor(N, a, shots=1024):
    qc, up_reg, down_reg, aux_bit, n = initialize_shor_circuit(N)
    apply_controlled_exponentiation(qc, up_reg, down_reg, aux_bit, a, N, n)
    result = measure_and_analyze(qc, up_reg, shots)
    for bitstring in result:
        x = int(bitstring, 2)
        r = find_period(x, 2 * n, N, a)
        if r:
            factor1 = math.gcd(pow(a, r//2) - 1, N)
            factor2 = math.gcd(pow(a, r//2) + 1, N)
            if factor1 not in [1, N] and factor2 not in [1, N]:
                print(factor1, factor2)
                return
    print(-1)