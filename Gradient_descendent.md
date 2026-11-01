```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the function and its derivative
def f(x):
    return (x + 3) ** 2

def grad_f(x):
    return 2 * (x + 3)

# 2. Initialize parameters
x_current = 2
learning_rate = 0.1
tolerance = 1e-6
max_iterations = 100
history = [x_current]

# 3. Gradient descent loop
for i in range(max_iterations):
    x_next = x_current - learning_rate * grad_f(x_current)

    if abs(x_next - x_current) < tolerance:
        print(f"Converged after {i + 1} iterations.")
        break

    x_current = x_next
    history.append(x_current)
    print(f"Iteration {i + 1}: x = {x_current:.4f}, f(x) = {f(x_current):.4f}")

# 4. Print the result
print("Local minima at x =", round(x_current, 4))
print("Function value at local minima y =", round(f(x_current), 4))

# 5. Plotting
plt.plot(history, [f(val) for val in history], marker='o')
plt.xlabel("x values")
plt.ylabel("f(x)")
plt.title("Gradient Descent Convergence")
plt.grid()
plt.show()
```