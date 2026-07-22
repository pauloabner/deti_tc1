
from utils.abner_math import add_bias, dot_product, plot_decision_boundary


# This module contains tests for the add_bias function in abner_math.py
# The add_bias function takes a list x and an optional bias value (default is 1) and returns a new list with the bias value added at the beginning of the list.
def test_add_bias():
    x = [2, -1, 0, 5]
    bias = [3, 1, 0, -2]
    for i in range(len(bias)):
        assert add_bias(x, bias[i]) == [bias[i]] + x
    assert add_bias(x) == [1] + x
    print("All tests passed!")  

def test_dot_product():
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    assert dot_product(v1, v2) == 32
    try:
        dot_product([1, 2], [1])
    except ValueError as e:
        assert str(e) == "Vectors must be of the same length"
    print("All tests passed!")

def test_plot_decision_boundary():
    w = [1, -1]
    b = 3
    plot_decision_boundary(w, b, title="Test Decision Boundary")
    print("All tests passed!")


if __name__ == "__main__":
    test_add_bias()
    test_dot_product()
    test_plot_decision_boundary()