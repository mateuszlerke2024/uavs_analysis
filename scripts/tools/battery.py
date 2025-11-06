from numpy import array
import pandas as pd
from scipy.interpolate import CubicSpline
from typing import Union

_default_temp_array = array([-40, -20, -10, 0, 25, 40, 55, 60])  # deg C
_interpolate = lambda cap, t: CubicSpline(_default_temp_array, cap)(t) / 100


def _get_eff_from_temp_li_ir_ph(temp_c):
    """Effective capacity for lithium iron phosphorate cells"""
    cap = array([46.6, 74.8, 88.1, 97.6, 100, 110.9, 104.4, 99.1])
    return _interpolate(cap, temp_c)


def _get_eff_from_temp_li_man(temp_c):
    """Effective capacity for lithium manganate cells"""
    cap = array([36.8, 68.0, 78.4, 97.6, 100, 101.2, 123.5, 110.3])
    return _interpolate(cap, temp_c)


def _get_eff_from_temp_li_cb_ox(temp_c):
    """Effective capacity for lithium cobalt oxide cells"""
    cap = array([11.7, 45.2, 73.6, 93.4, 100, 97.7, 99.6, 98.6])
    return _interpolate(cap, temp_c)


def _get_eff_from_temp_aver(temp_c):
    """Effective capacity, averaged"""
    ir = _get_eff_from_temp_li_ir_ph(temp_c)
    man = _get_eff_from_temp_li_man(temp_c)
    cb = _get_eff_from_temp_li_cb_ox(temp_c)
    return (ir + man + cb) / 3.0


def _get_eff_from_temp(temp_c):
    """
    Returns the effective capacity (as a percentage of nominal capacity at 25°C)
    for a given temperature in Celsius using a quadratic approximation.

    Parameters:
    temp_c (float): Temperature in degrees Celsius
    """
    return _get_eff_from_temp_aver(temp_c)


def joules2SoC(initial_percentage: float, usage: Union[pd.Series, array], temp: float = 25,
               nominal_capacity: float = 4500, voltage: float = 22.2, wear_capacity_coefficient: float = 0.8
               ) -> pd.Series:
    """
    Converts the energy usage from Joules to battery state-of-charge.

    Parameters:
        initial_percentage (float): Battery percentage before the flight.
        usage (Series | np.array): Energy used during the flight, in Joules.
        temp (float): Ambient temperature during the flight, in degrees Celsius.
        nominal_capacity (float): Nominal battery capacity, in mAh.
        voltage (float): Nominal battery voltage, in volts.
        wear_capacity_coefficient (float): Battery wear coefficient, found empirically.

    Returns:
        Series: Battery state-of-charge during the flight.
    """
    effective_capacity: float = nominal_capacity * voltage * 3.6  # From mAh to J
    effective_capacity *= _get_eff_from_temp(temp) * wear_capacity_coefficient  # Include temperature and wear effects
    initial_energy: float = effective_capacity * initial_percentage
    final_energy: pd.Series = initial_energy - usage
    final_soc: pd.Series = final_energy / effective_capacity
    return final_soc
