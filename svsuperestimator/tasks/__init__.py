"""svSuperEstimator's problem subpackage.

Contains the different optimization problems.
"""
from .blood_vessel_tuning import BloodVesselTuning
from .windkessel_tuning import WindkesselTuning
from .map_zero_d_result_to_three_d import MapZeroDResultToThreeD
from .map_three_d_result_on_centerline import MapThreeDResultOnCenterline
from .three_d_simulation import ThreeDSimulation
from .multi_fidelity_tuning import MultiFidelityTuning

__all__ = ["BloodVesselTuning"]

_task_mapping = {
    BloodVesselTuning.__name__: BloodVesselTuning,
    WindkesselTuning.__name__: WindkesselTuning,
    MapZeroDResultToThreeD.__name__: MapZeroDResultToThreeD,
    ThreeDSimulation.__name__: ThreeDSimulation,
    MapThreeDResultOnCenterline.__name__: MapThreeDResultOnCenterline,
    MultiFidelityTuning.__name__: MultiFidelityTuning,
}

VALID_TASKS = list(_task_mapping.keys())


def get_task_by_name(name):
    return _task_mapping[name]
