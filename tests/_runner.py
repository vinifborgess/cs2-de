"""Tiny dependency-free test runner: calls every ``test_*`` function in a namespace."""


def run(namespace: dict) -> int:
    failed = 0
    for name, fn in sorted((n, f) for n, f in namespace.items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0
