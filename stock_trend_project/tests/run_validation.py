from analytics.validation import validate_tests

if __name__ == "__main__":
    for name, ok in validate_tests():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")