from qiskit_ibm_runtime import QiskitRuntimeService

try:
    service = QiskitRuntimeService()   # uses saved credentials if present
    print("IBM Quantum setup is working.")
    print("Active account:", service.active_account())
    print("Active instance:", service.active_instance())

    backends = service.backends(simulator=False, operational=True)
    print("Available real backends:")
    for b in backends[:10]:
        print(" -", b.name)

except Exception as e:
    print("IBM Quantum is not set up correctly yet.")
    print("Error:", repr(e))