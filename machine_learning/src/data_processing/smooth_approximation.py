import functools
from dataclasses import dataclass

import numpy as np
import qpsolvers


class QuadraticApproximationError(Exception):
    pass


@dataclass
class SmoothMatrixApproximator:
    matrix_side_length: int = 6
    solver: str = "cvxopt"
    outlier_weight: float = 0.0001
    use_log_scale: bool = False

    def __call__(self, matrix: np.ndarray) -> np.ndarray:
        assert matrix.shape == (self.matrix_shape)
        solution = self._get_most_fitting_solution(matrix=matrix)
        return solution.x.reshape(matrix.shape)

    def _get_most_fitting_solution(
        self,
        matrix: np.ndarray,
    ) -> qpsolvers.solution.Solution:
        increasing_solution = self._get_increasing_solution(matrix=matrix)
        decreasing_solution = self._get_decreasing_solution(matrix=matrix)
        return (
            increasing_solution
            if self._get_solution_objective(increasing_solution, matrix)
            < self._get_solution_objective(decreasing_solution, matrix)
            else decreasing_solution
        )

    def _get_increasing_solution(
        self,
        matrix: np.ndarray,
    ) -> qpsolvers.solution.Solution:
        return self._solve_problem(
            self._get_increasing_quadratic_problem(matrix=matrix),
        )

    def _get_decreasing_solution(
        self,
        matrix: np.ndarray,
    ) -> qpsolvers.solution.Solution:
        return self._solve_problem(
            self._get_decreasing_quadratic_problem(matrix=matrix),
        )

    def _solve_problem(
        self,
        problem: qpsolvers.problem.Problem,
    ) -> qpsolvers.solution.Solution:
        solution = qpsolvers.solve_problem(problem, solver=self.solver)
        if not solution.found:
            msg = "Failed to converge"
            raise QuadraticApproximationError(msg)
        if self.use_log_scale:
            solution.x = 10**solution.x
        return solution

    def _get_decreasing_quadratic_problem(
        self,
        matrix: np.ndarray,
    ) -> qpsolvers.problem.Problem:
        problem = self._get_increasing_quadratic_problem(matrix=matrix)
        problem.G = -problem.G
        return problem

    def _get_increasing_quadratic_problem(
        self,
        matrix: np.ndarray,
    ) -> qpsolvers.problem.Problem:
        matrix_weights = self._get_matrix_weights(matrix=matrix)

        if self.use_log_scale:
            matrix = np.log10(matrix)
            matrix[np.isnan(matrix)] = 0

        return qpsolvers.problem.Problem(
            P=self.P * matrix_weights.reshape(-1),
            q=-2 * matrix.reshape(-1).astype("double") * matrix_weights.reshape(-1),
            G=self.G_increasing,
            h=self.h,
            lb=self.lb,
        )

    def _get_matrix_weights(self, matrix: np.ndarray) -> np.ndarray:
        is_outlier = np.isclose(matrix, 0)
        return is_outlier * self.outlier_weight + (1 - is_outlier) * 1

    @functools.cached_property
    def P(self) -> np.ndarray:  # noqa: N802
        return 2 * np.eye(self.num_variables)

    @functools.cached_property
    def G_increasing(self) -> np.ndarray:  # noqa: N802
        return np.concatenate([self.G_increasing_rows, self.G_increasing_columns])

    @property
    def G_increasing_columns(self) -> np.ndarray:  # noqa: N802
        is_current_variable = np.diag(np.ones(self.num_variables))
        is_one_column_above = np.diag(np.ones(self.num_variables - 1), k=1)
        differences_matrix = is_current_variable - is_one_column_above
        row_indices = np.arange(len(differences_matrix))
        row_indices_to_delete = row_indices[
            self.matrix_side_length - 1 :: self.matrix_side_length
        ]  # delete every 6th row
        row_indices_to_keep = np.array(
            [idx for idx in row_indices if idx not in row_indices_to_delete],
        )
        return differences_matrix[row_indices_to_keep]

    @property
    def G_increasing_rows(self) -> np.ndarray:  # noqa: N802
        is_current_variable = np.diag(np.ones(self.num_variables))
        is_one_row_above = np.diag(
            np.ones(self.num_variables - self.matrix_side_length),
            k=self.matrix_side_length,
        )
        differences_matrix = is_current_variable - is_one_row_above
        return differences_matrix[: -self.matrix_side_length]

    def _get_solution_objective(
        self,
        solution: qpsolvers.solution.Solution,
        matrix: np.ndarray,
    ) -> float:
        return np.linalg.norm(solution.x.reshape(6, 6) - matrix)

    @functools.cached_property
    def h(self) -> np.ndarray:
        return np.zeros(len(self.G_increasing))

    @functools.cached_property
    def lb(self) -> np.ndarray:
        return np.zeros(self.num_variables)

    @property
    def num_variables(self) -> int:
        return self.matrix_side_length**2

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return (self.matrix_side_length, self.matrix_side_length)
