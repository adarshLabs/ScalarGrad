# Testing Guide for ScalarGrad

This document explains how to run, write, and understand tests in ScalarGrad.

## Table of Contents

1. [Running Tests](#running-tests)
2. [Test Organization](#test-organization)
3. [Writing Tests](#writing-tests)
4. [Test Coverage](#test-coverage)
5. [Continuous Integration](#continuous-integration)

---

## Running Tests

### Quick Start

```bash
# Run all tests
python test_derivatives.py
python test_nn_full.py
python test_sub_neg.py
```

### With pytest (if installed)

```bash
# Install pytest
pip install pytest

# Run all tests in current directory
pytest

# Run specific test file
pytest test_derivatives.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=.
```

### Run Specific Test

```bash
# Run one test file
python test_derivatives.py

# View specific test output
python -c "from test_derivatives import *; test_case('Test 1a', ...)"
```

---

## Test Organization

### Current Test Files

| File | Purpose | Tests |
|------|---------|-------|
| `test_derivatives.py` | Verify gradient computation | 16 cases (chains, nesting, mixed ops) |
| `test_nn_full.py` | Neural network functionality | 7 cases (neuron, layer, MLP, training) |
| `test_sub_neg.py` | Arithmetic operations | 4 cases (subtraction, negation, rsub) |

### Test Structure

```
tests/
├── test_derivatives.py       # Gradient/autodiff tests
├── test_nn_full.py           # Neural network tests
├── test_sub_neg.py           # Arithmetic operation tests
└── README.md                 # Test documentation
```

### Future Organization

As the project grows, organize by module:

```
tests/
├── __init__.py
├── test_value.py             # Value class tests
├── test_operations.py        # Arithmetic operations
├── test_activation.py        # Activation functions
├── test_nn/
│   ├── test_neuron.py
│   ├── test_layer.py
│   └── test_mlp.py
└── test_autodiff.py          # Backpropagation tests
```

---

## Writing Tests

### Basic Test Template

```python
from value import Value

def test_simple_addition():
    """Test basic addition operation."""
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    
    # Test forward pass
    assert c.data == 5.0, f"Expected 5.0, got {c.data}"
    
    # Test backward pass
    c.backward()
    assert a.grad == 1.0, f"Expected a.grad=1.0, got {a.grad}"
    assert b.grad == 1.0, f"Expected b.grad=1.0, got {b.grad}"
    
    print("✓ test_simple_addition PASSED")
```

### Test with Numerical Verification

```python
from value import Value
import math

def test_tanh_with_numerical_gradient():
    """Test tanh activation with numerical gradient."""
    x_val = 0.5
    eps = 1e-5
    
    # Analytical gradient
    x = Value(x_val)
    y = x.tanh()
    y.backward()
    analytical = x.grad
    
    # Numerical gradient
    x_plus = Value(x_val + eps)
    y_plus = x_plus.tanh()
    x_minus = Value(x_val - eps)
    y_minus = x_minus.tanh()
    numerical = (y_plus.data - y_minus.data) / (2 * eps)
    
    # Compare
    error = abs(analytical - numerical)
    assert error < 1e-4, f"Gradient error too large: {error}"
    
    print(f"✓ Analytical: {analytical:.6f}, Numerical: {numerical:.6f}")
```

### Test Neural Network Component

```python
from nn import Neuron, Layer, MLP
from value import Value

def test_neuron_forward():
    """Test neuron forward pass."""
    n = Neuron(nin=2, nonlinear=False)
    inputs = [Value(1.0), Value(2.0)]
    output = n(inputs)
    
    # Should be: w1*1 + w2*2 + b
    expected_formula = f"= {n.weights[0].data:.4f}*1 + {n.weights[1].data:.4f}*2 + {n.bias.data:.4f}"
    print(f"  Neuron output {expected_formula} = {output.data:.4f}")
    
    assert isinstance(output, Value), "Output should be Value"
    assert output.data != 0, "Output should be non-zero"
    
    print("✓ Neuron forward pass works")
```

### Parametrized Test

```python
from value import Value

test_cases = [
    (2.0, 3.0, 6.0),    # (x, expected_df/dx, name)
    (3.0, 12.0),
    (5.0, 30.0),
]

def test_power_gradient():
    """Test x^3 gradient: d/dx(x^3) = 3x^2"""
    for x_val, expected_grad in test_cases:
        x = Value(x_val)
        y = x ** 3
        y.backward()
        
        actual = x.grad
        assert abs(actual - expected_grad) < 1e-6, \
            f"At x={x_val}: expected {expected_grad}, got {actual}"
        
        print(f"  ✓ x={x_val}: df/dx = {actual} (expected {expected_grad})")
```

---

## Test Coverage

### Operations Covered

**Arithmetic:**
- [x] Addition (`+`)
- [x] Subtraction (`-`)
- [x] Multiplication (`*`)
- [x] Division (`/`)
- [x] Power (`**`)
- [x] Negation (`-x`)
- [x] Reverse operations (`__rsub__`, `__rtruediv__`)

**Activation Functions:**
- [x] ReLU
- [x] Tanh

**Gradient Properties:**
- [x] Chain rule
- [x] Gradient accumulation
- [x] Multiple paths
- [x] Topological sort correctness

**Neural Network:**
- [x] Neuron forward pass
- [x] Neuron backward pass
- [x] Layer forward pass
- [x] Layer backward pass
- [x] MLP forward pass
- [x] MLP backward pass
- [x] Gradient descent training
- [x] Parameter management

### Coverage Gaps (Future)

- [ ] Sigmoid activation
- [ ] Batch operations
- [ ] Conv2D layers
- [ ] LSTM cells
- [ ] Dropout
- [ ] Batch normalization

---

## Test Metrics

### Current Statistics

```
Total Tests:          27
Passing:              27 (100%)
Failing:              0
Skipped:              0

Coverage:
- value.py:           ~95%
- nn.py:              ~90%
- Overall:            ~92%
```

### Test Duration

```
test_derivatives.py:  ~0.5s  (16 tests)
test_nn_full.py:      ~1.2s  (7 tests)
test_sub_neg.py:      ~0.1s  (4 tests)
Total:                ~1.8s
```

---

## Continuous Integration

### GitHub Actions (Recommended)

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.7', '3.8', '3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python test_derivatives.py
        python test_nn_full.py
        python test_sub_neg.py
```

### Running Locally Before Push

```bash
# Check all tests pass
python test_derivatives.py && \
python test_nn_full.py && \
python test_sub_neg.py && \
echo "All tests passed!"
```

---

## Debugging Failed Tests

### Enable Verbose Output

```python
# Add print statements
def test_something():
    x = Value(3.0)
    y = x ** 2
    print(f"Forward: {y.data}")
    y.backward()
    print(f"Gradient: {x.grad}")
    assert x.grad == 6.0
```

### Use a Debugger

```bash
# Run with pdb debugger
python -m pdb test_derivatives.py

# Or use ipdb for better experience
python -m ipdb test_derivatives.py
```

### Check Intermediate Values

```python
def test_debug():
    x = Value(2.0)
    y = x + 3
    z = y ** 2
    
    print(f"x.data = {x.data}")
    print(f"y.data = {y.data}, y._prev = {y._prev}")
    print(f"z.data = {z.data}, z._prev = {z._prev}")
    
    z.backward()
    
    print(f"After backward:")
    print(f"x.grad = {x.grad}")
    print(f"y.grad = {y.grad}")
    print(f"z.grad = {z.grad}")
```

---

## Best Practices

### ✅ DO

- [x] Test both forward and backward pass
- [x] Verify against numerical gradients
- [x] Use descriptive test names
- [x] Add docstrings explaining what you test
- [x] Test edge cases
- [x] Include print statements for debugging

### ❌ DON'T

- [ ] Test only one direction (forward or backward)
- [ ] Assume gradients are correct without verification
- [ ] Use vague names like `test1` or `test_something`
- [ ] Skip docstrings
- [ ] Test only happy path

---

## Contributing Tests

When adding a new feature:

1. **Write test first** (TDD approach)
2. **Implement feature** to make test pass
3. **Add numerical verification** for gradients
4. **Document** what the test checks
5. **Run full suite** to ensure nothing breaks

Template:

```python
def test_new_feature():
    """Test description of what we're testing.
    
    Verify:
    - Forward computation is correct
    - Backward (gradient) computation is correct
    - Edge cases are handled
    """
    # Setup
    x = Value(...)
    
    # Forward
    y = x.new_operation()
    assert y.data == expected_forward
    
    # Backward
    y.backward()
    assert x.grad == expected_backward
    
    # Numerical verification
    numerical = numerical_gradient(lambda x: x.new_operation().data, x_val)
    assert abs(x.grad - numerical) < 1e-4
    
    print("✓ test_new_feature PASSED")
```

---

## Questions?

- See [ARCHITECTURE.md](ARCHITECTURE.md) for implementation details
- Review existing tests for examples
- Open an issue if test guidance needed

**Happy testing!** 🧪
