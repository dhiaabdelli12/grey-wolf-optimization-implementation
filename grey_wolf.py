import numpy as np
import matplotlib.pyplot as plt


class Wolf:
    def __init__(self, sspace_dim, objective_function, bounds, load):
        self.A = 2 * np.repeat(2, sspace_dim) * np.random.uniform(0, 1) - np.repeat(2, sspace_dim)
        self.C = 2 * np.random.uniform(0, 1)
        self.position = self._init_position(sspace_dim, bounds)
        self.fitness = objective_function.evaluate(self.position)
        self.sspace_dim = sspace_dim
        self.obj_fn = objective_function
        self.bounds = bounds
        self.load = load

    def _compute_D(self, dominant_position, C):
        return np.abs(C * dominant_position - self.position)

    def _compute_X(self, dominant_position, A, D):
        return dominant_position - A * D

    def evaluate_fitness(self):
        return self.obj_fn.evaluate(self.position)

    def update_coeffs(self, a):
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        self.A = 2 * a * r1 - a
        self.C = 2 * r2

    def adjust_to_constraints(self):
        if self.bounds is not None:
            self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
        if self.load is not None:
            self.position[:-3] = self.load - (
                self.position[-3] + self.position[-2] + self.position[-1]
            )
            self.position[self.position < 0] = 0

    def update_position(self, positions, As, Cs):
        X_1 = positions[0] - As[0] * self._compute_D(positions[0], Cs[0])
        X_2 = positions[1] - As[1] * self._compute_D(positions[1], Cs[1])
        X_3 = positions[2] - As[2] * self._compute_D(positions[2], Cs[2])
        return (X_1 + X_2 + X_3) / 3

    def _init_position(self, sspace_dim, bounds):
        return np.random.uniform(bounds[0][0], bounds[1][0], sspace_dim)
    


class GWO:
    def __init__(
        self,
        sspace_dim,
        nb_wolves,
        max_iters,
        objective_function,
        bounds,
        load,
        minimize=True,
    ):
        self.sspace_dim = sspace_dim
        self.bounds = bounds
        self.load = load
        self.obj_fn = objective_function
        self.nb_wolves = nb_wolves
        self.max_iters = max_iters
        self.wolves = self._init_wolves()
        self.alpha = self.wolves[0]
        self.beta = self.wolves[1]
        self.delta = self.wolves[2]
        self.a = np.repeat(2, sspace_dim)

        self.minimize = minimize

        self.history = []

    def _init_wolves(self):
        wolves = [
            Wolf(self.sspace_dim, self.obj_fn, self.bounds, self.load)
            for _ in range(self.nb_wolves)
        ]
        sorted_wolves = sorted(
            wolves, key=lambda wolf: self.obj_fn.evaluate(wolf.position)
        )
        return sorted_wolves

    def adjust_to_contraints(self, position):
        if self.bounds is not None:
            position = np.clip(position, self.bounds[0], self.bounds[1])
        if self.load is not None:
            position[:-3] = self.load - (
                position[-3] + position[-2] + position[-1]
            )
            position[position < 0] = 0
        return position

    def _sort_wolves(self):
        self.wolves = sorted(
            self.wolves,
            key=lambda wolf: self.obj_fn.evaluate(wolf.position),
            reverse=True,
        )

    def _update_hierarchy(self):
        self._sort_wolves()
        self.alpha = self.wolves[0]
        self.beta = self.wolves[1]
        self.delta = self.wolves[2]



    def _step(self):
        for wolf in self.wolves:
            new_position = wolf.update_position(
                [self.alpha.position, self.beta.position, self.delta.position],
                [self.alpha.A, self.beta.A, self.delta.A],
                [self.alpha.C, self.beta.C, self.delta.C],
            )
            new_position = self.adjust_to_contraints(new_position)
            new_fitness = self.obj_fn.evaluate(new_position)
            if new_fitness < wolf.fitness:
                wolf.position = new_position
                wolf.fitness = new_fitness

            wolf.update_coeffs(self.a)

    def run(self):
        print("Executing GWO algorithm")
        print(f"Number of wolves: \t{self.nb_wolves}")
        print(f"Number of Iterations: \t{self.max_iters}")
        print("\n")
        self.history = []
        for idx in range(self.max_iters):
            self.a = 2 * (1 - idx / self.max_iters)
            print(
                f"\tStep {idx}/{self.max_iters}:\t alpha position: [{self.alpha.position[0]:.2f}...{self.alpha.position[30]:.2f}...{self.alpha.position[-1]:.2f}] alpha fitness: {self.alpha.fitness:.2f}"
            )
            self._step()
            self._update_hierarchy()

            self.history.append(
                {
                    "alpha_position": self.alpha.position,
                    "alpha_fitness": self.alpha.fitness,
                }
            )

    def results(self):
        print(f"GWO Run Results:\n")
        print(
            f"\tOptimal PV size: {self.alpha.position[-2]:.2f}\t\t Interval: [{self.bounds[0][-2]:.2f},{self.bounds[1][-2]:.2f}]"
        )
        print(
            f"\tOptimal ESS Capacity: {self.alpha.position[-1]:.2f}\t Interval: [{self.bounds[0][-1]:.2f},{self.bounds[1][-1]:.2f}]"
        )
        print(
            f"\tOptimal SF Capacity: {self.alpha.position[-3]:.2f}\t Interval: [{self.bounds[0][-3]:.2f},{self.bounds[1][-3]:.2f}]"
        )
        print(
            f"\tOptimal values for PD: \t\tInterval: [{self.bounds[0][0]:.2f},{self.bounds[1][0]:.2f}]\n"
        )
        for idx, value in enumerate(self.alpha.position[:-3]):
            loads = f"\t\tLoad: {self.load[idx]:.4f}" if self.load is not None else ""
            msg = f"\t\t PD_{idx+1}: {value:.4f}" + loads
            print(msg)

    def plot_run(self):
        x = range(self.max_iters)
        y = [hist["alpha_fitness"] for hist in self.history]
        _ = plt.figure()
        plt.plot(x, y)
        plt.xlabel("Step Number")
        plt.ylabel("Fitness of alpha")
        plt.title("GWO Run plot.")
        plt.show()

    def plot_wolf_movement(self):
        ...

    def __str__(self):
        msg = f"GWO Algorithm:\n"
        msg += f"\tNumber of wolves: {self.nb_wolves}\n"
        msg += f"\tMaximum Number of Iterations: {self.max_iters}\n"
        msg += f"\tsolution Space dimension: {self.bounds.shape[1]}\n"
        msg += "\tObjective: Minimize\n" if self.minimize else "Objective: Maximize\n"
        msg += "\n\tDominant Wolves: \n"
        msg += f"\t\t|Alpha:\tInitial Fitness: {self.alpha.fitness}\t Initial Position: [{self.alpha.position[0]:.4f} ... {self.alpha.position[-1]:.4f}]\n"
        msg += f"\t\t|Beta:\tInitial Fitness: {self.beta.fitness}\t Initial Position: [{self.beta.position[0]:.4f} ... {self.beta.position[-1]:.4f}]\n"
        msg += f"\t\t|Delta:\tInitial Fitness: {self.delta.fitness}\t Initial Position: [{self.delta.position[0]:.4f} ... {self.delta.position[-1]:.4f}]\n"
        return msg