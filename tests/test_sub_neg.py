from scalargrad.value import Value

print("Testing __sub__, __rsub__, __neg__")
print("=" * 60)

# Test 1: Subtraction
print("\n1. __sub__: f(x) = x - 2")
x = Value(5.0)
result = x - 2
result.backward()
print(f"   x = {x.data}, f(x) = {result.data}, df/dx = {x.grad}")
print(f"   Expected: df/dx = 1 (derivative of x - 2 is 1)")
assert x.grad == 1.0, "FAIL"
print("   ✓ PASS")

# Test 2: Reverse Subtraction
print("\n2. __rsub__: f(x) = 10 - x")
x = Value(3.0)
result = 10 - x  # Uses __rsub__
result.backward()
print(f"   x = {x.data}, f(x) = {result.data}, df/dx = {x.grad}")
print(f"   Expected: df/dx = -1 (derivative of 10 - x is -1)")
assert x.grad == -1.0, "FAIL"
print("   ✓ PASS")

# Test 3: Negation
print("\n3. __neg__: f(x) = -x")
x = Value(7.0)
result = -x
result.backward()
print(f"   x = {x.data}, f(x) = {result.data}, df/dx = {x.grad}")
print(f"   Expected: df/dx = -1 (derivative of -x is -1)")
assert x.grad == -1.0, "FAIL"
print("   ✓ PASS")

# Test 4: Complex with negation
print("\n4. Complex: f(x) = -(x ** 2)")
x = Value(3.0)
result = -(x ** 2)
result.backward()
print(f"   x = {x.data}, f(x) = {result.data}, df/dx = {x.grad}")
print(f"   Expected: df/dx = -2*3 = -6 (derivative of -x^2 is -2x)")
assert abs(x.grad - (-6.0)) < 1e-9, f"FAIL: got {x.grad}"
print("   ✓ PASS")

print("\n" + "=" * 60)
print("All tests PASS!")
