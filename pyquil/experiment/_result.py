##############################################################################
# Copyright 2016-2019 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################
"""
Schema definition of an ExperimentResult, which encapsulates the outcome of a collection of
measurements that are aimed at estimating the expectation value of some observable.
"""
import logging
import sys
import warnings
from typing import List, Optional, Union

import numpy as np

from pyquil.experiment._setting import ExperimentSetting

if sys.version_info < (3, 7):
    from pyquil.external.dataclasses import dataclass
else:
    from dataclasses import dataclass


log = logging.getLogger(__name__)


def bitstrings_to_expectations(
        bitstrings: np.ndarray,
        correlations: Optional[List[List[int]]] = None
) -> np.ndarray:
    """
    Given a list of bitstrings (each of which is represented as a list of bits), map them to
    expectation values and return the desired correlations. If no correlations are given, then just
    the 1 -> -1, 0 -> 1 mapping is performed.

    :param bitstrings: List of bitstrings to map.
    :param correlations: Correlations to calculate. Defaults to None, which is equivalent to the
        list [[0], [1], ..., [n-1]] for bitstrings of length n.
    :return: A list of expectation values, of the same length as the list of bitstrings. The
        "width" could be different than the length of an individual bitstring (n) depending on
        the value of the ``correlations`` parameter.
    """
    expectations: np.ndarray = 1 - 2 * bitstrings

    if correlations is None:
        return expectations

    region_size = len(expectations[0])

    e = []
    for c in correlations:
        where = np.zeros(region_size, dtype=bool)
        where[c] = True
        e.append(np.prod(expectations[:, where], axis=1))
    return np.stack(e, axis=-1)


@dataclass(frozen=True)
class ExperimentResult:
    """An expectation and standard deviation for the measurement of one experiment setting
    in a tomographic experiment.

    In the case of readout error calibration, we also include
    expectation, standard deviation and count for the calibration results, as well as the
    expectation and standard deviation for the corrected results.
    """

    setting: ExperimentSetting
    expectation: Union[float, complex]
    total_counts: int
    std_err: Union[float, complex] = None
    raw_expectation: Union[float, complex] = None
    raw_std_err: float = None
    calibration_expectation: Union[float, complex] = None
    calibration_std_err: Union[float, complex] = None
    calibration_counts: int = None
    correlations: Optional[List['ExperimentResult']] = None

    def __init__(self, setting: ExperimentSetting,
                 expectation: Union[float, complex],
                 total_counts: int,
                 stddev: Union[float, complex] = None,
                 std_err: Union[float, complex] = None,
                 raw_expectation: Union[float, complex] = None,
                 raw_stddev: float = None,
                 raw_std_err: float = None,
                 calibration_expectation: Union[float, complex] = None,
                 calibration_stddev: Union[float, complex] = None,
                 calibration_std_err: Union[float, complex] = None,
                 calibration_counts: int = None,
                 correlations: Optional[List['ExperimentResult']] = None):

        object.__setattr__(self, 'setting', setting)
        object.__setattr__(self, 'expectation', expectation)
        object.__setattr__(self, 'total_counts', total_counts)
        object.__setattr__(self, 'raw_expectation', raw_expectation)
        object.__setattr__(self, 'calibration_expectation', calibration_expectation)
        object.__setattr__(self, 'calibration_counts', calibration_counts)
        object.__setattr__(self, 'correlations', correlations)

        if stddev is not None:
            warnings.warn("'stddev' has been renamed to 'std_err'")
            std_err = stddev
        object.__setattr__(self, 'std_err', std_err)

        if raw_stddev is not None:
            warnings.warn("'raw_stddev' has been renamed to 'raw_std_err'")
            raw_std_err = raw_stddev
        object.__setattr__(self, 'raw_std_err', raw_std_err)

        if calibration_stddev is not None:
            warnings.warn("'calibration_stddev' has been renamed to 'calibration_std_err'")
            calibration_std_err = calibration_stddev
        object.__setattr__(self, 'calibration_std_err', calibration_std_err)

    def get_stddev(self) -> Union[float, complex]:
        warnings.warn("'stddev' has been renamed to 'std_err'")
        return self.std_err

    def set_stddev(self, value: Union[float, complex]):
        warnings.warn("'stddev' has been renamed to 'std_err'")
        object.__setattr__(self, 'std_err', value)

    stddev = property(get_stddev, set_stddev)

    def get_raw_stddev(self) -> float:
        warnings.warn("'raw_stddev' has been renamed to 'raw_std_err'")
        return self.raw_std_err

    def set_raw_stddev(self, value: float):
        warnings.warn("'raw_stddev' has been renamed to 'raw_std_err'")
        object.__setattr__(self, 'raw_std_err', value)

    raw_stddev = property(get_raw_stddev, set_raw_stddev)

    def get_calibration_stddev(self) -> Union[float, complex]:
        warnings.warn("'calibration_stddev' has been renamed to 'calibration_std_err'")
        return self.calibration_std_err

    def set_calibration_stddev(self, value: Union[float, complex]):
        warnings.warn("'calibration_stddev' has been renamed to 'calibration_std_err'")
        object.__setattr__(self, 'calibration_std_err', value)

    calibration_stddev = property(get_calibration_stddev, set_calibration_stddev)

    def __str__(self):
        return f'{self.setting}: {self.expectation} +- {self.std_err}'

    def __repr__(self):
        return f'ExperimentResult[{self}]'

    def serializable(self):
        return {
            'type': 'ExperimentResult',
            'setting': self.setting,
            'expectation': self.expectation,
            'std_err': self.std_err,
            'total_counts': self.total_counts,
            'raw_expectation': self.raw_expectation,
            'raw_std_err': self.raw_std_err,
            'calibration_expectation': self.calibration_expectation,
            'calibration_std_err': self.calibration_std_err,
            'calibration_counts': self.calibration_counts,
        }


def ratio_variance(a: Union[float, np.ndarray],
                   var_a: Union[float, np.ndarray],
                   b: Union[float, np.ndarray],
                   var_b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Given random variables 'A' and 'B', compute the variance on the ratio Y = A/B. Denote the
    mean of the random variables as a = E[A] and b = E[B] while the variances are var_a = Var[A]
    and var_b = Var[B] and the covariance as Cov[A,B]. The following expression approximates the
    variance of Y

    Var[Y] \approx (a/b) ^2 * ( var_a /a^2 + var_b / b^2 - 2 * Cov[A,B]/(a*b) )

    We assume the covariance of A and B is negligible, resting on the assumption that A and B
    are independently measured. The expression above rests on the assumption that B is non-zero,
    an assumption which we expect to hold true in most cases, but makes no such assumptions
    about A. If we allow E[A] = 0, then calculating the expression above via numpy would complain
    about dividing by zero. Instead, we can re-write the above expression as

    Var[Y] \approx var_a /b^2 + (a^2 * var_b) / b^4

    where we have dropped the covariance term as noted above.

    See the following for more details:
      - https://doi.org/10.1002/(SICI)1097-0320(20000401)39:4<300::AID-CYTO8>3.0.CO;2-O
      - http://www.stat.cmu.edu/~hseltman/files/ratio.pdf
      - https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables

    :param a: Mean of 'A', to be used as the numerator in a ratio.
    :param var_a: Variance in 'A'
    :param b: Mean of 'B', to be used as the numerator in a ratio.
    :param var_b: Variance in 'B'
    """
    return var_a / b**2 + (a**2 * var_b) / b**4


def corrected_experiment_result(
        result: ExperimentResult,
        calibration: ExperimentResult,
        correlations: Optional[List[ExperimentResult]] = None,
) -> ExperimentResult:
    """

    :param raw:
    :param calibration:
    :return:
    """
    corrected_expectation = result.expectation / calibration.expectation
    corrected_variance = ratio_variance(result.expectation,
                                        result.std_err ** 2,
                                        calibration.expectation,
                                        calibration.std_err ** 2)
    return ExperimentResult(setting=result.setting,
                            expectation=corrected_expectation,
                            std_err=np.sqrt(corrected_variance).item(),
                            total_counts=result.total_counts,
                            raw_expectation=result.expectation,
                            raw_std_err=result.std_err,
                            calibration_expectation=calibration.expectation,
                            calibration_std_err=calibration.std_err,
                            calibration_counts=calibration.total_counts,
                            correlations=correlations)


def apply_readout_correction(
        results: List[ExperimentResult],
        calibrations: List[ExperimentResult]
) -> List[ExperimentResult]:
    """

    :param results:
    :param calibrations:
    :return:
    """
    corrected_results = []
    for result, cal in zip(results, calibrations):

        corrs = []
        for r, c in zip(result.correlations, cal.correlations):
            corrs.append(corrected_experiment_result(r, c))

        corrected_results.append(corrected_experiment_result(result, cal, corrs))

    return corrected_results


def results_to_dict(results: List[ExperimentResult]) -> dict:
    """

    :param results:
    :return:
    """
    results_dict = {}
    for result in results:
        out_operator = result.setting.out_operator.id(sort_ops=False)
        result_dict = result.serializable()
        for key, value in result_dict.items():
            results_dict[f'{out_operator}_{key}'] = value
        if result.correlations:
            for c in result.correlations:
                oo = c.setting.out_operator.id(sort_ops=False)
                rd = c.serializable()
                for key, value in rd.items():
                    results_dict[f'{oo}_{key}'] = value
    return results_dict
