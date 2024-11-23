import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from gurobipy import Model, GRB, LinExpr, QuadExpr

class Parallel_Benders:
    """
    Solves MILPs of the following form using Benders decomposition:

    min c_T*x + f_T*y     over x,y
    s.t   A*x + B*y >= b (m constraints)
          D*y >= d       (n constraints)
          x >= 0         (Nx-dimensional vector)
          y >= 0         (Ny-dimensional vector)
          y integer
    """

    def __init__(self):
        # Data
        self.Nx = 2                                        # Size of vector x
        self.Ny = 1                                        # Size of vector y
        self.m = 2                                         # Number of complicating constraints A*x + B*y >= b
        self.n = 1                                         # Number of constraints D*y >= d
        self.c = np.array([-4, -3]).reshape((self.Nx, 1))
        self.f = np.array([-5]).reshape((self.Ny, 1))
        self.A = np.array([[-2, -3], [-2, -1]]).reshape((self.m, self.Nx))
        self.B = np.array([-1, -3]).reshape((self.m, self.Ny))
        self.b = np.array([-12, -12]).reshape((self.m, 1))
        self.D = np.array([-1]).reshape((self.n, self.Ny))
        self.d = np.array([-20]).reshape((self.n, 1))

        self.y_init = np.array([10], dtype=int).reshape((self.Ny, 1))  # Initial feasible guess (Important!)

        self.eps = 1e-3                                    # Convergence value
        self.max_iterations = 100                           # Number of maximum iterations
        self.LB = sys.float_info.min                       # Lower bound of objective function
        self.UB = sys.float_info.max                       # Upper bound of objective function
        self.optimality_cuts = []
        self.feasibility_cuts = []
        self.lower_bounds = []
        self.upper_bounds = []

    def solve_problem(self):
        """Solves the MILP using Benders decomposition with parallel subproblem solving."""
        i = 0
        y_sol = self.y_init
        obj_value_master = None
        y_candidates = [y_sol + np.random.randint(-10, 10, size=(self.Ny, 1)) for _ in range(4)]  # Example perturbations
        print(y_candidates)
        while abs((self.UB - self.LB) / self.UB) >= self.eps and i <= self.max_iterations:
            # Generate multiple y_sol candidates for parallel subproblem solving
            
            # Solve subproblems in parallel
            results = []
            with ProcessPoolExecutor() as executor:
                future_to_y = {
                    executor.submit(self.solve_subproblem, y): y for y in y_candidates
                }

                for future in as_completed(future_to_y):
                    results.append((future.result(), future_to_y[future]))

            # Process results from subproblems
            for (p, obj_value_sp, sp_status, _), y in results:
                if sp_status == GRB.OPTIMAL:  # Subproblem feasible
                    self.optimality_cuts.append(p)
                    self.UB = min(self.UB, np.dot(self.f.T, y).item() + obj_value_sp)
                else:  # Subproblem unbounded
                    r = self.solve_modified_subproblem(y)
                    self.feasibility_cuts.append(r)

            # Solve Master problem
            temp_y_sol, temp_obj_value_master = self.solve_master_problem()

            if temp_y_sol is None and temp_obj_value_master is None:
                i+=1
                continue
            else:
                y_sol = temp_y_sol
                obj_value_master = temp_obj_value_master

            # Update lower bound
            self.LB = obj_value_master

            # Update iteration index
            i += 1

            # Save values for plotting
            self.lower_bounds.append(self.LB)
            self.upper_bounds.append(self.UB)

            print(f"Iteration {i}: UB={self.UB}, LB={self.LB}, y_sol={y_sol}")
            # if abs((self.UB - self.LB) / self.UB) >= self.eps is not True:
            #     sol = y_sol

        if y_sol is None:
            print("\nThe algorithm did not converge in the given iterations.")
            return
        print(y_sol)
        # Solve sub-problem with the optimal y solution to get the optimal x solution
        _, _, _, x_sol = self.solve_subproblem(y_sol)

        # Display the results
        self.show_results(i, obj_value_master, x_sol, y_sol)

    def show_results(self, i, obj_value_master, x_sol, y_sol):
        """Displays the results of the optimization problem."""
        if i > self.max_iterations:
            print("\nThe algorithm did not converge in the given iterations.")
        else:
            print("\n*** Optimal solution to the MILP problem found. ***")
        print(f"The optimal value is: {obj_value_master}")

        if x_sol is not None:
            print(f"The optimal solution is x*={x_sol.flatten()}, y*={y_sol.flatten()}")
        else:
            print("\nThe algorithm did not find the optimal solution. Please try another initial feasible guess y_init!")

    def solve_subproblem(self, y):
        """Solves the sub-problem in the dual form using Gurobi."""
        model = Model("Subproblem")
        model.setParam("OutputFlag", 0)

        # Dual variables
        p = model.addMVar(shape=self.m, lb=0, name="p")

        # Objective
        model.setObjective(p @ (self.b.flatten() - self.B @ y.flatten()), GRB.MAXIMIZE)

        # Constraints
        model.addConstr(p @ self.A <= self.c.flatten(), "dual_constraint")

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return p.X, model.objVal, model.status, None  # x_sol not directly available in dual form
        else:
            return None, None, model.status, None

    def solve_modified_subproblem(self, y):
        """Solves the modified sub-problem in the dual form using Gurobi."""
        model = Model("ModifiedSubproblem")
        model.setParam("OutputFlag", 0)

        # Dual variables
        p = model.addMVar(shape=self.m, lb=0, name="p")

        # Constraints
        model.addConstr(p @ (self.b.flatten() - self.B @ y.flatten()) == 1, "ray_constraint")
        model.addConstr(p @ self.A <= 0, "dual_constraint")

        # Objective
        model.setObjective(0, GRB.MAXIMIZE)

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return p.X
        else:
            return None

    def solve_master_problem(self):
        """Solves the Master problem using Gurobi."""
        model = Model("MasterProblem")
        model.setParam("OutputFlag", 0)

        # Variables
        y = model.addMVar(shape=self.Ny, vtype=GRB.INTEGER, lb=0, name="y")
        n = model.addVar(lb=-GRB.INFINITY, name="n")

        # Objective
        model.setObjective(self.f.flatten() @ y + n, GRB.MINIMIZE)

        # Constraints
        model.addConstr(self.D @ y >= self.d.flatten(), "master_constraint")

        for p in self.optimality_cuts:
            model.addConstr(n >= p @ (self.b.flatten() - self.B @ y), "opt_cut")
        for r in self.feasibility_cuts:
            model.addConstr(0 >= r @ (self.b.flatten() - self.B @ y), "feas_cut")

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return y.X.reshape((self.Ny, 1)), model.objVal
        else:
            return None, None
            # raise RuntimeError("Master problem did not converge!")


# Usage
if __name__ == "__main__":
    benders = Parallel_Benders()
    benders.solve_problem()
