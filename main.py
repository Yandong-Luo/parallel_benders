from benders import Benders
from parallel_benders import Parallel_Benders
from plotter import plot_convergence


def main():
    """Solves the (MILP) optimization problem."""
    benders_dec = Benders()
    benders_dec = Parallel_Benders()
    # benders_dec.solve_problem()
    plot_convergence(benders_dec.upper_bounds, benders_dec.lower_bounds)


if __name__ == '__main__':
    main()
