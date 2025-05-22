def apply_gate(self, gate, targets):
    full_gate = 1
    for i in range(self.num_qubits):
        if i in targets:
            full_gate = np.kron(full_gate, gate)
        else:
            full_gate = np.kron(full_gate, np.eye(2))
    self.state = full_gate @ self.state