from scalargrad.value import Value
import math

print("=" * 80)
print("COMPREHENSIVE TEST CASES: CHAINS, NESTED EXPRESSIONS & MANUAL DERIVATIVES")
print("=" * 80)

def numerical_gradient(func, x_val, eps=1e-5):
    """Compute numerical gradient using finite differences"""
    x_plus = Value(x_val + eps)
    x_minus = Value(x_val - eps)
    f_plus = func(x_plus).data
    f_minus = func(x_minus).data
    return (f_plus - f_minus) / (2 * eps)

def test_case(name, func, x_val, analytical_grad):
    """Test a case and compare analytical vs numerical gradients"""
    print(f"\n{name}")
    print("-" * 80)
    
    x = Value(x_val)
    result = func(x)
    result.backward()
    
    numerical = numerical_gradient(func, x_val)
    analytical = x.grad
    
    print(f"  Input: x = {x_val}")
    print(f"  Output: f(x) = {result.data:.6f}")
    print(f"  Analytical gradient (df/dx) = {analytical:.6f}")
    print(f"  Numerical gradient (df/dx)  = {numerical:.6f}")
    print(f"  Expected gradient           = {analytical_grad:.6f}")
    print(f"  Error: {abs(analytical - numerical):.2e}")
    
    if abs(analytical - numerical) < 1e-4 and abs(analytical - analytical_grad) < 1e-4:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
    
    return analytical, numerical

# ============================================================================
# TEST 1: Simple Power Function
# ============================================================================
print("\n\n1. SIMPLE POWER FUNCTION")
print("=" * 80)

# f(x) = x^2, at x=3: df/dx = 2*3 = 6
test_case(
    "Test 1a: f(x) = x^2 at x=3",
    lambda x: x ** 2,
    3.0,
    6.0  # 2*x = 2*3
)

# f(x) = x^3, at x=2: df/dx = 3*x^2 = 3*4 = 12
test_case(
    "Test 1b: f(x) = x^3 at x=2",
    lambda x: x ** 3,
    2.0,
    12.0  # 3*x^2 = 3*4
)

# ============================================================================
# TEST 2: Chains of Multiplication
# ============================================================================
print("\n\n2. CHAINS OF MULTIPLICATION")
print("=" * 80)

# f(x) = x * x * x = x^3, at x=2: df/dx = 3*x^2 = 12
test_case(
    "Test 2a: f(x) = x * x * x (chain mult) at x=2",
    lambda x: x * x * x,
    2.0,
    12.0
)

# f(x) = x * 2 * 3 = 6x, at x=5: df/dx = 6
test_case(
    "Test 2b: f(x) = x * 2 * 3 (constant chain) at x=5",
    lambda x: x * 2 * 3,
    5.0,
    6.0
)

# ============================================================================
# TEST 3: Chains of Addition
# ============================================================================
print("\n\n3. CHAINS OF ADDITION")
print("=" * 80)

# f(x) = x + x + x = 3x, at x=4: df/dx = 3
test_case(
    "Test 3a: f(x) = x + x + x at x=4",
    lambda x: x + x + x,
    4.0,
    3.0
)

# f(x) = (x + 2) + (x + 3) = 2x + 5, at x=1: df/dx = 2
test_case(
    "Test 3b: f(x) = (x + 2) + (x + 3) at x=1",
    lambda x: (x + 2) + (x + 3),
    1.0,
    2.0
)

# ============================================================================
# TEST 4: NESTED EXPRESSIONS
# ============================================================================
print("\n\n4. NESTED EXPRESSIONS")
print("=" * 80)

# f(x) = (x + 2)^2, at x=1: df/dx = 2(x+2) = 2*3 = 6
test_case(
    "Test 4a: f(x) = (x + 2)^2 at x=1",
    lambda x: (x + 2) ** 2,
    1.0,
    6.0  # 2*(x+2) = 2*3
)

# f(x) = (x + 2)^3, at x=0: df/dx = 3(x+2)^2 = 3*4 = 12
test_case(
    "Test 4b: f(x) = (x + 2)^3 at x=0",
    lambda x: (x + 2) ** 3,
    0.0,
    12.0  # 3*(x+2)^2 = 3*4
)

# ============================================================================
# TEST 5: Complex Nested Operations
# ============================================================================
print("\n\n5. COMPLEX NESTED OPERATIONS")
print("=" * 80)

# f(x) = (x^2 + 1) * (x + 1)
# df/dx = (2x)(x+1) + (x^2+1)(1) = 2x^2 + 2x + x^2 + 1 = 3x^2 + 2x + 1
# at x=2: df/dx = 3*4 + 2*2 + 1 = 12 + 4 + 1 = 17
test_case(
    "Test 5a: f(x) = (x^2 + 1) * (x + 1) at x=2",
    lambda x: (x ** 2 + 1) * (x + 1),
    2.0,
    17.0  # 3x^2 + 2x + 1 = 12 + 4 + 1
)

# f(x) = ((x + 1)^2) * x
# df/dx = 2(x+1) * x + (x+1)^2 * 1 = 2x(x+1) + (x+1)^2
# at x=2: df/dx = 2*2*3 + 9 = 12 + 9 = 21
test_case(
    "Test 5b: f(x) = ((x + 1)^2) * x at x=2",
    lambda x: ((x + 1) ** 2) * x,
    2.0,
    21.0  # 2x(x+1) + (x+1)^2 = 12 + 9
)

# ============================================================================
# TEST 6: Division Operations
# ============================================================================
print("\n\n6. DIVISION OPERATIONS")
print("=" * 80)

# f(x) = x / 2, at x=4: df/dx = 1/2 = 0.5
test_case(
    "Test 6a: f(x) = x / 2 at x=4",
    lambda x: x / 2,
    4.0,
    0.5
)

# f(x) = 6 / x, at x=2: df/dx = -6/x^2 = -6/4 = -1.5
test_case(
    "Test 6b: f(x) = 6 / x at x=2",
    lambda x: Value(6) / x,
    2.0,
    -1.5  # -6/x^2 = -6/4
)

# ============================================================================
# TEST 7: Mixed Operations (Add, Mul, Div, Pow)
# ============================================================================
print("\n\n7. MIXED OPERATIONS")
print("=" * 80)

# f(x) = (x^2 + x) / (x + 2), at x=1
# Let u = x^2 + x, du/dx = 2x + 1 = 3
# Let v = x + 2, dv/dx = 1
# df/dx = (du/dx * v - u * dv/dx) / v^2 = (3*3 - 2*1) / 9 = 7/9 ≈ 0.7778
test_case(
    "Test 7a: f(x) = (x^2 + x) / (x + 2) at x=1",
    lambda x: (x ** 2 + x) / (x + 2),
    1.0,
    7.0 / 9.0  # (7/9)
)

# f(x) = (x + 1)^2 / x, at x=2
# u = (x+1)^2, du/dx = 2(x+1) = 6
# v = x, dv/dx = 1
# df/dx = (6*2 - 9*1) / 4 = 3/4 = 0.75
test_case(
    "Test 7b: f(x) = (x + 1)^2 / x at x=2",
    lambda x: (x + 1) ** 2 / x,
    2.0,
    0.75
)

# ============================================================================
# TEST 8: Multiple Uses of Same Variable (Gradient Accumulation)
# ============================================================================
print("\n\n8. GRADIENT ACCUMULATION (Multiple Uses)")
print("=" * 80)

# f(x) = x + x^2 + x^3, at x=2
# df/dx = 1 + 2x + 3x^2 = 1 + 4 + 12 = 17
test_case(
    "Test 8a: f(x) = x + x^2 + x^3 at x=2",
    lambda x: x + x ** 2 + x ** 3,
    2.0,
    17.0  # 1 + 2*2 + 3*4
)

# f(x) = (x * x) + (x * x), at x=3
# This is 2x^2, df/dx = 4x = 12
test_case(
    "Test 8b: f(x) = (x * x) + (x * x) [2x^2] at x=3",
    lambda x: (x * x) + (x * x),
    3.0,
    12.0  # 4*3
)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("All test cases compare:")
print("  1. Analytical gradients (from backprop)")
print("  2. Numerical gradients (finite differences)")
print("  3. Manual derivatives (hand-calculated)")
print("=" * 80)
