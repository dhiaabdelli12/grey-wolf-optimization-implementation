import numpy as np


class ObjectiveFunction:
    """Objective function class definition"""

    def __init__(
        self,
        a,
        b,
        p_d_rated,
        fuel_price,
        c_capital_pv,
        c_replacement_pv,
        crf_pv,
        c_capital_ess,
        c_replacement_ess,
        c_capital_sf,
        c_replacement_sf,
        crf_sf,
        crf_ess,
        em_fuel,
    ) -> None:
        self.pd_values = None
        self.p_pv = None
        self.e_ess = None
        self.e_sf = None
        self.a = a
        self.b = b
        self.p_d_rated = p_d_rated
        self.fuel_price = fuel_price
        self.c_capital_pv = c_capital_pv
        self.c_replacement_pv = c_replacement_pv
        self.crf_pv = crf_pv
        self.c_capital_ess = c_capital_ess
        self.c_replacement_ess = c_replacement_ess
        self.c_capital_sf = c_capital_sf
        self.c_replacement_sf = c_replacement_sf
        self.crf_ess = crf_ess
        self.em_fuel = em_fuel

        self.crf_sf = crf_sf

    def _fuel_cost(self) -> float:
        """Calculates the fuel cost."""
        return np.sum(
            self.fuel_price
            * (
                self.pd_values * self.a
                + np.repeat(self.p_d_rated * self.b, self.pd_values.size)
            )
        )

    def _pv_cost(self) -> float:
        """Calculates the photovoltaic panels cost."""
        return (self.c_capital_pv + self.c_replacement_pv) * self.p_pv

    def _ess_cost(self) -> float:
        """Calculates Energy Storage System (ESS) cost."""
        return (self.c_capital_ess + self.c_replacement_ess) * self.e_ess

    def _sf_cost(self) -> float:
        """Calculates Stack fuel battery cost"""
        return (self.c_capital_sf + self.c_replacement_sf) * self.e_sf

    def _cost(self) -> float:
        """Calculates the total cost."""
        return (
            self._fuel_cost()
            + self._pv_cost() * self.crf_pv
            + self._ess_cost() * self.crf_ess
            + self._sf_cost() * self.crf_sf
        )

    def _emission(self) -> float:
        """Calculates the gas emissions."""
        return np.sum(
            self.em_fuel
            * (
                self.pd_values * self.a
                + np.repeat(self.p_d_rated * self.b, self.pd_values.size)
            )
        )

    def evaluate(self, inputs: np.ndarray) -> float:
        """Calculates the total cost of ESS and PV with the gas emissions."""
        self.pd_values = inputs[:-3]
        self.e_sf = inputs[-3]
        self.p_pv = inputs[-2]
        self.e_ess = inputs[-1]
        return self._cost() + self._emission()


def get_bounds(
    nb_hours: int,
    pd_min: float,
    pd_max: float,
    pv_min: float,
    pv_max: float,
    e_ess_min: float,
    e_ess_max: float,
    e_sf_min: float,
    e_sf_max: float,
):
    """Get the bounds for optimization variables from pre-determined values."""
    lower_bounds = np.append(np.repeat(pd_min, nb_hours), [e_sf_min, pv_min, e_ess_min])
    upper_bounds = np.append(np.repeat(pd_max, nb_hours), [e_sf_max, pv_max, e_ess_max])
    bounds = np.vstack((lower_bounds, upper_bounds))
    return bounds


def get_crf(r: float, y: int):
    """Calculates the Captial Recovery Factor (CRF) given an interest rate and a life span"""
    return (r * (1 + r) ** y) / ((1 + r) ** y - 1)
