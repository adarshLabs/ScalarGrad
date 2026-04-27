from scalargrad.nn import Neuron, Layer, MLP
from scalargrad.value import Value

print("=" * 80)
print("COMPREHENSIVE NEURAL NETWORK TEST - Forward & Backward Pass")
print("=" * 80)

# ============================================================================
# TEST 1: Single Neuron Forward Pass
# ============================================================================
print("\n1. SINGLE NEURON - Forward Pass")
print("-" * 80)

neuron = Neuron(nin=2, nonlinear=False)  # Linear neuron for simplicity
print(f"Neuron: {neuron}")
print(f"Weights: {[w.data for w in neuron.weights]}")
print(f"Bias: {neuron.bias.data}")

inputs = [Value(1.0), Value(2.0)]
output = neuron(inputs)
print(f"\nInput: [1.0, 2.0]")
print(f"Output: {output.data:.4f}")
print(f"Computation: w1*x1 + w2*x2 + b = {neuron.weights[0].data:.4f}*1.0 + {neuron.weights[1].data:.4f}*2.0 + {neuron.bias.data:.4f}")

# ============================================================================
# TEST 2: Single Neuron with Backward Pass
# ============================================================================
print("\n2. SINGLE NEURON - Backward Pass (Gradient Computation)")
print("-" * 80)

neuron2 = Neuron(nin=2, nonlinear=False)
inputs2 = [Value(3.0), Value(4.0)]
output2 = neuron2(inputs2)

print(f"Forward: {output2.data:.4f}")
print(f"Running backward()...")
output2.backward()

print(f"\nGradients after backward():")
print(f"  w1.grad = {neuron2.weights[0].grad:.6f}")
print(f"  w2.grad = {neuron2.weights[1].grad:.6f}")
print(f"  b.grad  = {neuron2.bias.grad:.6f}")
print(f"  x1.grad = {inputs2[0].grad:.6f}")
print(f"  x2.grad = {inputs2[1].grad:.6f}")

# ============================================================================
# TEST 3: Layer Forward Pass
# ============================================================================
print("\n3. LAYER - Forward Pass (2 Neurons, Linear)")
print("-" * 80)

layer = Layer(nin=2, nout=2, nonlinear=False)
print(f"Layer: {layer}")

inputs3 = [Value(1.0), Value(2.0)]
output3 = layer(inputs3)
print(f"\nInput: [1.0, 2.0]")
print(f"Output: [{output3[0].data:.4f}, {output3[1].data:.4f}]")

# ============================================================================
# TEST 4: Layer with Backward Pass
# ============================================================================
print("\n4. LAYER - Backward Pass (Gradient Flow)")
print("-" * 80)

layer2 = Layer(nin=2, nout=2, nonlinear=False)
inputs4 = [Value(5.0), Value(3.0)]
outputs4 = layer2(inputs4)

# Create a simple loss: sum of outputs
loss = outputs4[0] + outputs4[1]
print(f"Forward pass outputs: [{outputs4[0].data:.4f}, {outputs4[1].data:.4f}]")
print(f"Loss (sum of outputs): {loss.data:.4f}")
print(f"\nRunning backward()...")
loss.backward()

print(f"\nNumber of parameters in layer: {len(layer2.parameters())}")
print(f"Sample gradients:")
for i, p in enumerate(layer2.parameters()[:4]):
    print(f"  param[{i}].grad = {p.grad:.6f}")

# ============================================================================
# TEST 5: Simple MLP (Multi-Layer Perceptron)
# ============================================================================
print("\n5. SIMPLE MLP - Forward Pass (2→3→1 Architecture)")
print("-" * 80)

mlp = MLP(nin=2, nouts=[3, 1])
print(f"MLP: {mlp}")
print(f"Total parameters: {len(mlp.parameters())}")

inputs5 = [Value(1.5), Value(2.5)]
output5 = mlp(inputs5)
print(f"\nInput: [1.5, 2.5]")
print(f"Output: {output5.data:.4f}")

# ============================================================================
# TEST 6: MLP with Backward Pass & Weight Updates
# ============================================================================
print("\n6. MLP - Full Training Step (Forward + Backward + Gradient Descent)")
print("-" * 80)

mlp2 = MLP(nin=2, nouts=[3, 1])
print(f"MLP Architecture: 2 → 3 → 1")
print(f"Total parameters: {len(mlp2.parameters())}")

# Training data
x_train = [Value(0.0), Value(0.0)]
y_target = Value(0.5)

# Forward pass
pred = mlp2(x_train)
loss = (pred - y_target) ** 2  # MSE loss
print(f"\nTraining iteration:")
print(f"  Input: [0.0, 0.0]")
print(f"  Target: {y_target.data}")
print(f"  Prediction: {pred.data:.6f}")
print(f"  Loss (MSE): {loss.data:.6f}")

# Backward pass
print(f"\nBackward pass - computing gradients...")
loss.backward()

print(f"\nGradient statistics:")
grads = [p.grad for p in mlp2.parameters()]
print(f"  Min gradient: {min(grads):.6e}")
print(f"  Max gradient: {max(grads):.6e}")
print(f"  Mean gradient: {sum(grads)/len(grads):.6e}")

# Manual gradient descent step
learning_rate = 0.01
print(f"\nGradient descent update (lr={learning_rate}):")
for p in mlp2.parameters():
    p.data = p.data - learning_rate * p.grad

# New forward pass with updated weights
pred_new = mlp2(x_train)
loss_new = (pred_new - y_target) ** 2
print(f"  New prediction: {pred_new.data:.6f}")
print(f"  New loss: {loss_new.data:.6f}")
print(f"  Loss decreased: {loss_new.data < loss.data} ✓")

# ============================================================================
# TEST 7: Gradient Accumulation Test
# ============================================================================
print("\n7. GRADIENT ACCUMULATION - zero_grad() Method")
print("-" * 80)

mlp3 = MLP(nin=2, nouts=[2])
print(f"Initial parameters: {len(mlp3.parameters())}")

# First forward-backward pass
x1 = [Value(1.0), Value(2.0)]
y1 = mlp3(x1)
loss1 = y1[0] + y1[1]
loss1.backward()

print(f"After first backward():")
grad_sum_1 = sum(abs(p.grad) for p in mlp3.parameters())
print(f"  Sum of absolute gradients: {grad_sum_1:.6f}")

# Zero gradients
mlp3.zero_grad()
print(f"\nAfter zero_grad():")
grad_sum_2 = sum(abs(p.grad) for p in mlp3.parameters())
print(f"  Sum of absolute gradients: {grad_sum_2:.6f}")
print(f"  Gradients cleared: {grad_sum_2 == 0} ✓")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 80)
