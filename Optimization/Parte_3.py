import numpy as np
from matplotlib import use
use('TkAgg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

def plot_surface(func, x_domain, y_domain):
    """
    Plots the surface of a given mathematical function over the specified domain.

    Parameters:
    - func: a symbolic function defined using sympy.
    - x_domain: tuple, the domain range for x (e.g., (-10, 10)).
    - y_domain: tuple, the domain range for y (e.g., (-10, 10)).
    """
    # Convert symbolic function to a lambda function for evaluation
    x, y = sp.symbols('x y')
    func_lambda = sp.lambdify((x, y), func, modules=['numpy'])
    
    # Create a grid of points
    x_vals = np.linspace(x_domain[0], x_domain[1], 100)
    y_vals = np.linspace(y_domain[0], y_domain[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = func_lambda(X, Y)
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    ax.set_title(f'Surface plot of the function: {sp.pretty(func)}')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    plt.show(block=True)

def plot_surface_with_gradients(func, x_domain, y_domain, points):
    """
    Plots the surface of a function and overlays the gradient vectors at specified points.

    Parameters:
    - func: sympy expression, the function to plot.
    - x_domain, y_domain: tuples, the range of x and y axes.
    - points: list of tuples, the points where gradients are evaluated and plotted.
    """
    # Convert symbolic function to a lambda function for evaluation
    x, y = sp.symbols('x y')
    func_lambda = sp.lambdify((x, y), func, modules=['numpy'])
    
    # Create a grid of points
    x_vals = np.linspace(x_domain[0], x_domain[1], 100)
    y_vals = np.linspace(y_domain[0], y_domain[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = func_lambda(X, Y)
    
    # Compute the gradient
    gradient = compute_gradient(func, [x, y])
    print(gradient)

    # Evaluate gradient at specified points
    gradients_at_points = evaluate_gradient_at_points(gradient, points)
    print(gradients_at_points)

    # Plotting the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

    # Overlay the gradient vectors
    for i, point in enumerate(points):
        grad = gradients_at_points[i]
        norm = np.linalg.norm(grad)
        unit_vector = grad / norm if norm != 0 else grad

        ax.quiver(point[0], point[1], float(func_lambda(point[0], point[1])),
                  unit_vector[0], unit_vector[1], 0,
                  color='r', length=1, normalize=True)

    ax.set_title(f'Surface plot of the function with Gradient Vectors')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    plt.show(block=True)

def compute_gradient(func, variables):
    """
    Computes the gradient vector of a given function.

    Parameters:
    - func: sympy expression, the function for which the gradient is computed.
    - variables: list of sympy symbols, the variables with respect to which the gradient is computed.

    Returns:
    - gradient: list of sympy expressions, the gradient vector.
    """
    # Compute the gradient by calculating the partial derivatives
    gradient = [sp.diff(func, var) for var in variables]
    
    return gradient

def evaluate_gradient_at_points(gradient, points):
    """
    Evaluates the gradient vector of a function at specified points.

    Parameters:
    - gradient: list of sympy expressions, the gradient vector.
    - points: list of tuples, each tuple is a point (x, y) where the gradient is evaluated.

    Returns:
    - evaluated_gradients: list of numpy arrays, each array is the evaluated gradient at a specific point.
    """
    x, y = sp.symbols('x y')
    evaluated_gradients = []
    for point in points:
        grad_at_point = np.array([float(grad.evalf(subs={x: point[0], y: point[1]})) for grad in gradient])
        evaluated_gradients.append(grad_at_point)
    return evaluated_gradients

# Example usage
x, y = sp.symbols('x y')


func1 = x**3 * y**2 + 1
points_func1 = [(0, 0), (7.4, -6.3)]
plot_surface_with_gradients(func1, (-10, 10), (-10, 10), points_func1)

func2 = sp.sin(x**2) + x * sp.cos(y**3)
points_func2 = [(1.5, -5.5), (-10, -10)]
plot_surface_with_gradients(func2, (-10, 10), (-10, 10), points_func2)

func3 = 3**(2*x) + 5**(4*y) + 2*x + y**4
points_func3 = [(-4, -2), (-2, 9)]
plot_surface_with_gradients(func3, (-10, 10), (-10, 10), points_func3)
