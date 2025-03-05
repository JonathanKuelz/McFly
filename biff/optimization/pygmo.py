from abc import ABC, ABCMeta, abstractmethod
import hashlib
import inspect
from typing import Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import pygmo as pg


class ReferenceCloneClass(type):
    """
    A metaclass that hides the reference_clone method of a class.
    """

    def __dir__(self):
        return [x for x in super().__dir__() if x != 'reference_clone']


class AbstractCloneableMetaClass(ReferenceCloneClass, ABCMeta):
    """This is necessary to define custom metaclasses on abstract classes."""
    pass


class UDP(ABC, metaclass=AbstractCloneableMetaClass):
    """
    We try to solve:
        minimize: fitness
        subject to: equality_constraints == 0
               and: inequality_constraints <= 0
    """

    @abstractmethod
    def _eval_fitness(self, x: np.ndarray) -> np.ndarray:
        """
        This method is mandatory.

        Returns the fitness value only.
        """
        pass

    def _eval_equality_constraints(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        This method is optional.

        Returns the equality constraints only.
        """
        pass

    def _eval_inequality_constraints(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        This method is optional.

        Returns the inequality constraints only.
        """
        pass

    @abstractmethod
    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """
        This method is also mandatory.

        Returns [lower_bounds], [upper_bounds].
        """
        pass

    def fitness(self, x: np.array) -> Union[Collection[float], np.array]:
        """
        pygmo always assumes that the fitness function returns fitness and constraints in the order (f, eq, ineq).

        To implement the interface, check out:
            - _eval_fitness
            - _eval_equality_constraints
            - _eval_inequality_constraints
        """
        f = self._eval_fitness(x)
        eq = self._eval_equality_constraints(x)
        ineq = self._eval_inequality_constraints(x)
        if eq is None:
            eq = np.array([])
        if ineq is None:
            ineq = np.array([])
        return np.hstack([f, eq, ineq])

    def get_name(self) -> str:
        """Returns the name of the problem."""
        return self.__class__.__name__

    def get_nec(self) -> int:
        """Returns the number of equality constraints."""
        return 0

    def get_nic(self) -> int:
        """Returns the number of inequality constraints."""
        return 0

    def reference_clone(self):
        """
        Creates a reference clone of `self`, making a new class that:
        - Explicitly redefines methods, preserving correct signatures.
        - Delegates all calls to the original instance.
        - Supports attribute access and inspection.

        :return: A new instance of a dynamically created class that mirrors `self`.
        """
        base_cls = type(self)
        this = self  # Capture the current instance

        # Collect all methods from the base class
        method_dict = {}

        for name, func in inspect.getmembers(base_cls, predicate=inspect.isfunction):
            # Define a wrapper that forwards the method call to `this`
            def make_delegate(method_name):
                return lambda self_, *args, **kwargs: getattr(this, method_name)(*args, **kwargs)
            method_dict[name] = make_delegate(name)

        def __init__(self_): pass
        method_dict["__init__"] = __init__

        # Define a dynamic class
        Clone = type("Clone", (base_cls,), method_dict)

        return Clone()


class UdpWithGradient(UDP, ABC):

    def __init__(self, stores_grad_during_fitness: bool = True):
        """
        A general template for UDPs that provide gradients.

        :param stores_grad_during_fitness: If True, the gradient is computed during the fitness evaluation and must
            be stored for later retrieval using _store_gradient. If false, the _get_gradient method must be implemented.
        """
        self.use_fitness_grad = stores_grad_during_fitness
        self._fitness_gradient_buffer: Dict[str, np.ndarray] = {}
        self._eq_gradient_buffer: Dict[str, np.ndarray] = {}
        self._ineq_gradient_buffer: Dict[str, np.ndarray] = {}

    def _store_gradient(self, x: np.ndarray, grad: np.ndarray, where: str):
        """
        Store the gradient computed in a fitness computation for later retrieval in the gradient function.

        Note: Numpy arrays cannot be hashed. Pythons hashing adds salt and is therefore not deterministic. This might
            be problematic for multiprocessing. The use of hashlib circumvents this.
        """
        hash_ = hashlib.sha256(x.tobytes()).hexdigest()
        buffer = {
            "fitness": self._fitness_gradient_buffer,
            "eq": self._eq_gradient_buffer,
            "ineq": self._ineq_gradient_buffer
        }
        buffer[where][hash_] = grad

    def _get_gradient(self, x: np.ndarray) -> np.ndarray:
        """Custom implementation to obtain the gradient of x."""
        raise NotImplementedError("Custom gradient retrieval not implemented.")

    def _get_buffered_gradient(self, hash_: str) -> np.ndarray:
        """Custom implementation to obtain the gradient of x."""
        grads = list()
        buffers = [self._fitness_gradient_buffer]
        if self.get_nec() > 0:
            buffers.append(self._eq_gradient_buffer)
        if self.get_nic() > 0:
            buffers.append(self._ineq_gradient_buffer)

        for buffer in buffers:
            if hash_ not in buffer:
                raise ValueError("Gradient not stored. Make sure to call _store_gradient during fitness evaluation.")
            grads.append(buffer[hash_])
        return np.hstack(grads)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the gradient of the fitness function.

        :param x: The input to the fitness function.
        :return: The gradient of the fitness function.
        """
        if not self.use_fitness_grad:
            return self._get_gradient(x)

        hash_ = hashlib.sha256(x.tobytes()).hexdigest()
        return self._get_buffered_gradient(hash_)


def optimize_udp(udp: UDP,
                 algo: str = 'auglag',
                 nlopt_local_algo: str = 'lbfgs',
                 x0: Optional[Collection[np.ndarray]] = None,
                 pop_size: int = 1,
                 verbosity: int = 50,
                 ) -> pg.population:
    """
    Shortcut to optimize a UDP with some commonly used options.

    :param udp: The UDP to optimize.
    :param algo: The algorithm to use.
    :param x0: The initial guess.
    :param pop_size: The population size.
    :param verbosity: The verbosity level.
    """
    algo = pg.algorithm(uda = pg.nlopt(algo))
    if nlopt_local_algo is not None:
        algo.extract(pg.nlopt).local_optimizer = pg.nlopt('lbfgs')
    algo.set_verbosity(verbosity)

    pop = pg.population(prob=udp, size=pop_size)
    if x0 is not None:
        for i, x in enumerate(x0):
            if not isinstance(x, np.ndarray):
                raise ValueError(f"Initial guess {x} at postion {i} is not a numpy array. Initial guess must be an"
                                 f"iterable with one guess for every individual, even if pop_size is 1.")
            pop.set_x(i, x)
    return algo.evolve(pop)


def udp_sgd(udp: UdpWithGradient,
            x0: np.ndarray,
            step_size: float = 1e-2,
            num_iter: int = 1000,
            lambda_constraint: float = 1e2,
            verbosity: int = 50
            ):
    """
    Stochastic gradient descent for UDPs.

    :param udp: The UDP to optimize.
    :param x0: The initial guess.
    :param step_size: The step size.
    :param num_iter: The number of iterations.
    :param lambda_constraint: The constraint multiplier.
    :param verbosity: The logging interval.
    """
    x = x0
    f0 = udp.fitness(x)
    neq = udp.get_nec()
    niq = udp.get_nic()
    nf = len(f0) - neq - niq
    dim = x0.shape[-1]
    for i in range(num_iter):
        vals = udp.fitness(x)
        f = vals[:nf]
        e = vals[nf:nf + udp.get_nec()].reshape(neq, 1)
        if niq != 0:
            raise NotImplementedError("Gradient descent not implemented for inequality constraints.")
        g = udp.gradient(x)
        gf = g[:dim * nf].reshape(-1, dim)
        ge = g[dim * nf:dim * (udp.get_nec() + nf)].reshape(-1, dim)

        update = step_size * gf.sum(axis=0)
        update = update + (step_size * ge * e * lambda_constraint).sum(axis=0)
        x = x - update

        if i % verbosity == 0:
            print(f"Iteration {i}: x = {x}, f = {f}, e = {e.flatten()}")
    return x