from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from typing import Callable
from sympy import *
import numpy as np
import time


class ButlerVolmer:
    def __init__(self, interval: list):
        self.interval = interval
        self.root = self.get_root(visualization=False)

    @staticmethod
    def f(x) -> Callable:
        """
        The Butler-Volmer equation in electrochemical processes relates
        the current density to the potential in an electrode.
        This function describes this process.
        """
        alpha = 0.2
        beta = 2

        return np.e ** (alpha * x) - np.e ** ((alpha - 1) * x) - beta

    def get_root(self, visualization: bool) -> None:
        """
        Find the root of a given f and plot its graph, showing where the root is located
        and confirming that the interval contains a root
        """
        start = self.interval[0]
        end = self.interval[-1]

        x = np.linspace(start, end, 1000)
        y = self.f(x)
        root = fsolve(self.f, 0)[0]
        print(f'The root of the Butler Volmer function is {root}')

        if visualization:
            Visualization.plot_function(x, y)
            Visualization.draw_point(root, 0)
            Visualization.show_graph()
        print(f'{"-" * 50}')


class Visualization:
    def __init__(self, f: ButlerVolmer):
        self.f = f

    def visualize_function(self, interval):
        start = interval[0]
        end = interval[-1]
        x = np.linspace(start, end, 1000)
        y = self.f(x)
        plt.clf()
        plt.grid()
        plt.plot(x, y)
        plt.show()

    @staticmethod
    def plot_function(x, y):
        plt.grid()
        plt.plot(x, y)

    @staticmethod
    def draw_point(x, y):
        plt.plot(x, y, 'go')

    @staticmethod
    def show_graph():
        plt.show()


class NumericalMethods:
    @staticmethod
    def numerical_method(func):
        def wrapper(*args, **kwargs):
            print(f'Performing {func.__name__} Method:')
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            print(f'Time to find root: {end - start}s')
            print(f'{"-" * 50}')

        return wrapper

    @staticmethod
    def difference(start, end):
        return abs(end - start)

    @staticmethod
    def center(interval_start, interval_end):
        return (interval_start + interval_end) / 2

    @staticmethod
    def is_negative(number):
        return number < 0

    @staticmethod
    def calculate_derivative_value(function, value):
        """
        Takes a derivative expression and calculates the value from a specified point
        """
        replaced_variable = str(function).replace('x', str(value))
        return eval(replaced_variable)

    @staticmethod
    def calculate_secant_line(f, interval_start, interval_end):
        start = interval_start
        end = interval_end
        return start - f(start) * (end - start) / (f(end) - f(start))

    @numerical_method
    def bisection(self, f: Callable, interval: list, tolerance: float, max_iterations: int, visualize: bool):
        # TODO - Pode usar a formula pra  descobrir o n maximo de iteracoes:
        # TODO - k > (log(b0 - a0) - log(erro)) / log(2)
        interval_start = interval[0]
        interval_end = interval[-1]

        current_iteration = 0
        center_point = None
        solution_found = True

        visualization = Visualization(f)

        while self.difference(interval_start, interval_end) >= tolerance:
            if current_iteration > max_iterations:
                solution_found = False
                break
            current_iteration += 1
            center_point = self.center(interval_start, interval_end)

            if self.is_negative(f(interval_start) * f(center_point)):
                interval_end = center_point
            else:
                interval_start = center_point

            if f(center_point) == 0.0:
                solution_found = True
                break

            if visualize:
                visualization.visualize_function([interval_start, interval_end])

        if solution_found:
            print(f'Solution found at {current_iteration} iterations, The x value is aprox. {center_point}')
        else:
            print(f'Solution not found')

        return center_point

    @numerical_method
    def newton(self, f: Callable, initial_guess: float, tolerance: float, max_iterations: int):
        x = Symbol('x')
        f_diff = f(x).diff(x)

        approximation = initial_guess
        for i in range(max_iterations):
            approximation_value = f(approximation)

            if abs(approximation_value) < tolerance:
                print(f'Found solution {approximation} at {i} iterations')
                print(f'{f(approximation)=}')
                return approximation
            derivative_value = self.calculate_derivative_value(f_diff, approximation_value)

            approximation = approximation - approximation_value / derivative_value
        print('Solution not found')

    @numerical_method
    def secant(self, f: Callable, interval: list[float], tolerance: float, max_iterations: int):
        interval_start = interval[0]
        interval_end = interval[-1]

        secant_value = None
        for i in range(max_iterations):
            if not self.is_negative(f(interval_start) * f(interval_end)):
                print(f'Root not contained on interval')
                break
            secant_value = self.calculate_secant_line(f, interval_start, interval_end)
            function_value_on_secant = f(secant_value)
            function_value_on_start = f(interval_start)
            function_value_on_end = f(interval_end)

            if function_value_on_secant == 0:
                print(f'Root found')
                break

            if self.is_negative(function_value_on_start * function_value_on_secant):
                interval_end = secant_value
            elif self.is_negative(function_value_on_end * function_value_on_secant):
                interval_start = secant_value

            if secant_value - self.calculate_secant_line(f, interval_start, interval_end) < tolerance:
                break
        print(f'{secant_value=}, {i=}')
        return self.calculate_secant_line(f, interval_start, interval_end)


def main():
    interval = [-10, 10]
    func = ButlerVolmer(interval)
    f = func.f

    NumericalMethods().bisection(f, interval, 0.01, 30, visualize=False)
    NumericalMethods().newton(f, 10, 0.01, 30)
    NumericalMethods().secant(f, interval, 0.01, 3000)


if __name__ == '__main__':
    main()
