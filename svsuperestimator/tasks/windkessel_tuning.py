"""This module holds the WindkesselTuning task."""
from __future__ import annotations

import os
import sys
import pdb
import pickle
from datetime import datetime
from multiprocessing import get_context
from typing import Any, Dict, Optional

import numpy as np
import orjson
import pandas as pd
import particles
import seaborn as sns
import matplotlib.pyplot as plt
import pysvzerod as svzerodplus
from particles import distributions as dists
from particles import smc_samplers as ssp
from rich.progress import BarColumn, Progress
from scipy import stats
from pysvzerod import Solver

from .. import reader, visualizer
from ..reader import utils as readutils
from . import plotutils, statutils, taskutils
from .task import Task


class WindkesselTuning(Task):
    """Windkessel tuning task.

    Tunes absolute resistance of Windkessel outles to mean outlet flow and
    minimum and maximum pressure at inlet targets.
    """

    TASKNAME = "windkessel_tuning"

    DEFAULTS = {
        "zerod_config_file": None,
        "num_procs": 1,
        "theta_obs": None,
        "theta_range": None,
        "y_obs": None,
        "num_particles": 100,
        "num_rejuvenation_steps": 2,
        "resampling_threshold": 0.5,
        "noise_factor": 0.05,
        "forward_model": None,
        **Task.DEFAULTS,
    }

    def core_run(self) -> None:
        """Core routine of the task."""

        self.theta_range = self.config["theta_range"]

        # Load the 0D simulation configuration
        zerod_config_handler = reader.SvZeroDSolverInputHandler.from_file(
            self.config["zerod_config_file"]
        )

        # Refine inflow boundary using cubic splines
        inflow_bc = zerod_config_handler.boundary_conditions["INFLOW"][
            "bc_values"
        ]
        inflow_bc["Q"] = taskutils.refine_with_cubic_spline(
            inflow_bc["Q"], zerod_config_handler.num_pts_per_cycle
        ).tolist()
        inflow_bc["t"] = np.linspace(
            inflow_bc["t"][0],
            inflow_bc["t"][-1],
            zerod_config_handler.num_pts_per_cycle,
        ).tolist()

        # Get ground truth distal to proximal ratio
        theta_obs = np.array(self.config["theta_obs"])
        self.log("Setting target parameters to:", theta_obs)
        self.database["theta_obs"] = theta_obs.tolist()

        # Setup forward model
        model = self.config["forward_model"]
        if model == "RC":
            self.forward_model = _Forward_ModelRC(zerod_config_handler)
        elif model == "RpRd":
            self.forward_model = _Forward_ModelRpRd(zerod_config_handler)
        else:
            raise NotImplementedError("Unknown forward model " + model)

        # Determine target observations through one forward evaluation
        y_obs = np.array(self.config["y_obs"])
        self.log("Setting target observation to:", y_obs)
        self.database["y_obs"] = y_obs.tolist()

        # Determine noise covariance
        std_vector = self.config["noise_factor"] * y_obs
        self.log("Setting std vector to:", std_vector)
        self.database["y_obs_std"] = std_vector.tolist()

        # Setup the iterator
        self.log("Setup tuning process")
        smc_runner = _SMCRunner(
            forward_model=self.forward_model,
            y_obs=y_obs,
            len_theta=len(theta_obs),
            likelihood_std_vector=std_vector,
            prior_bounds=self.config["theta_range"],
            num_procs=self.config["num_procs"],
            num_particles=self.config["num_particles"],
            resampling_strategy="systematic",
            resampling_threshold=self.config["resampling_threshold"],
            num_rejuvenation_steps=self.config["num_rejuvenation_steps"],
            console=self.console,
        )

        # Run the iterator
        self.log("Starting tuning process")
        all_particles, all_weights, all_logpost = smc_runner.run()
        self.database["particles"] = all_particles
        self.database["weights"] = all_weights
        self.database["logpost"] = all_logpost

        # Save parameters to file
        self.database["timestamp"] = datetime.now().strftime(
            "%m/%d/%Y, %H:%M:%S"
        )

    def post_run(self) -> None:
        """Postprocessing routine of the task."""

        results: Dict[str, Any] = {}

        # Read raw results
        self.log("Read raw result")
        particles = np.array(self.database["particles"][-1])
        weights = np.array(self.database["weights"][-1])
        log_post = np.array(self.database["logpost"][-1])
        zerod_config_handler = reader.SvZeroDSolverInputHandler.from_file(
            self.config["zerod_config_file"]
        )

        # Calculate metrics
        self.log("Calculate metrics")
        ground_truth = self.database["theta_obs"]
        wmean = statutils.particle_wmean(particles=particles, weights=weights)
        cov = statutils.particle_covmat(particles=particles, weights=weights)
        std = [cov[i][i] ** 0.5 for i in range(cov.shape[0])]
        wmean_error = [abs(m - gt) / gt for m, gt in zip(wmean, ground_truth)]

        max_post = statutils.particle_map(
            particles=particles, posterior=log_post
        )
        map_error = [abs(m - gt) / gt for m, gt in zip(max_post, ground_truth)]
        results["metrics"] = {
            "ground_truth": ground_truth,
            "weighted_mean": wmean,
            "weighted_mean_error": wmean_error,
            "weighted_std": std,
            "maximum_a_posteriori": max_post,
            "maximum_a_posteriori_error": map_error,
            "covariance_matrix": cov,
        }

        # Calculate exponential metrics
        particles_exp = np.exp(particles)
        ground_truth_exp = np.exp(self.database["theta_obs"])
        wmean_exp = statutils.particle_wmean(
            particles=particles_exp, weights=weights
        )
        maxap_exp = statutils.particle_map(
            particles=particles_exp, posterior=log_post
        )
        cov_exp = statutils.particle_covmat(
            particles=particles_exp, weights=weights
        )
        std_exp = [cov_exp[i][i] ** 0.5 for i in range(cov_exp.shape[0])]
        wmean_exp_error = [
            abs(m - gt) / gt for m, gt in zip(wmean_exp, ground_truth_exp)
        ]
        map_exp_error = [
            abs(m - gt) / gt for m, gt in zip(maxap_exp, ground_truth_exp)
        ]
        results["metrics"].update(
            {
                "exp_ground_truth": ground_truth_exp,
                "exp_weighted_mean": wmean_exp,
                "exp_weighted_mean_error": wmean_exp_error,
                "exp_weighted_std": std_exp,
                "exp_maximum_a_posteriori": maxap_exp,
                "exp_maximum_a_posteriori_error": map_exp_error,
                "exp_covariance_matrix": cov_exp,
            }
        )

        # Save the postprocessed result to a file
        self.log("Save postprocessed results")
        with open(
            os.path.join(self.output_folder, "results.json"), "wb"
        ) as ff:
            ff.write(
                orjson.dumps(
                    results,
                    option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY,
                )
            )

        for sample, name in zip([wmean, max_post], ["mean", "map"]):
            # set boundary conditions
            self.forward_model.evaluate(sample)

            # write configuration to file
            fn = os.path.join(self.output_folder, "solver_" + name + ".in")
            self.forward_model.to_file(fn)

            # write 0D results to file
            fn = os.path.join(self.output_folder, "solution_" + name + ".csv")
            self.forward_model.simulate_csv(fn)

    def generate_report(self) -> visualizer.Report:
        """Generate the task report."""

        # Add 3D plot of mesh with 0D elements
        report = visualizer.Report()
        report.add("Overview")
        branch_data = readutils.get_0d_element_coordinates(self.project)
        model_plot = plotutils.create_3d_geometry_plot_with_vessels(
            self.project, branch_data
        )
        report.add([model_plot])
        zerod_config_handler = reader.SvZeroDSolverInputHandler.from_file(
            self.config["zerod_config_file"]
        )
        num_pts_per_cycle = zerod_config_handler.num_pts_per_cycle
        bc_map = zerod_config_handler.vessel_to_bc_map

        result_map = pd.read_csv(
            os.path.join(self.output_folder, "solution_map.csv")
        )
        result_mean = pd.read_csv(
            os.path.join(self.output_folder, "solution_mean.csv")
        )

        # Read raw and postprocessed results
        with open(
            os.path.join(self.output_folder, "results.json"), "rb"
        ) as ff:
            results = orjson.loads(ff.read())
        particles = np.array(self.database["particles"][-1])
        weights = np.array(self.database["weights"][-1])
        zerod_config = reader.SvZeroDSolverInputHandler.from_file(
            self.config["zerod_config_file"]
        )

        # Format the labels
        outlet_bcs = zerod_config.outlet_boundary_conditions
        bc_names = list(outlet_bcs.keys())
        theta_names = [rf"$\theta_{i}$" for i in range(len(self.theta_range))]

        # Create parallel coordinates plot
        plot_range = [np.min(self.theta_range[:]), np.max(self.theta_range[:])]
        paracoords = visualizer.Plot2D()
        paracoords.add_parallel_coordinates_plots(
            particles.T,
            bc_names,
            color_by=weights,
            plotrange=plot_range,
        )

        # Add heatmap for the covariance
        cov_plot = visualizer.Plot2D(title="Covariance")
        cov_plot.add_heatmap_trace(
            x=bc_names,
            y=bc_names,
            z=results["metrics"]["covariance_matrix"],
            name="Covariance",
        )
        report.add([paracoords, cov_plot])

        map_opts: Dict[str, Any] = {
            "name": "MAP estimate",
            "showlegend": True,
            "color": "#EF553B",
            "width": 3,
        }
        mean_opts: Dict[str, Any] = {
            "name": "Mean estimate",
            "showlegend": True,
            "color": "#636efa",
            "width": 3,
            "dash": "dash",
        }

        # Calculate histogram data
        bandwidth = 0.02
        bins = int(self.config["num_particles"] / 10)

        # Create distribition plots for all boundary conditions
        n_dim = particles.shape[1]
        if n_dim == 2:
            report.add(f"Add bivariate results")
            joint_plot2(particles[:, 0], particles[:, 1], weights, self.config["theta_range"], "bivariate.png")
            
        for i in range(n_dim):
            report.add(f"Results for theta_{i}")

            # bins = int(
            #     (self.theta_range[i][1] - self.theta_range[i][0]) / bandwidth
            # )
            counts, bin_edges = np.histogram(
                particles[:, i],
                bins=bins,
                weights=weights,
                density=True,
                range=self.theta_range[i],
            )

            # Create kernel density estimation plot for BC
            distplot = visualizer.Plot2D(
                title="Weighted histogram and kernel density estimation",
                xaxis_title=theta_names[i],
                yaxis_title="Kernel density",
                xaxis_range=self.theta_range[i],
            )
            distplot.add_bar_trace(
                x=bin_edges,
                y=counts,
                name="Weighted histogram",
            )
            distplot.add_vline_trace(
                x=results["metrics"]["ground_truth"][i], text="Ground Truth"
            )
            gt = results["metrics"]["ground_truth"][i]
            wmean = results["metrics"]["weighted_mean"][i]
            std = results["metrics"]["weighted_std"][i]
            wmean_error = results["metrics"]["weighted_mean_error"][i] * 100
            map = results["metrics"]["maximum_a_posteriori"][i]
            map_error = (
                results["metrics"]["maximum_a_posteriori_error"][i] * 100
            )

            gt_exp = results["metrics"]["exp_ground_truth"][i]
            wmean_exp = results["metrics"]["exp_weighted_mean"][i]
            std_exp = results["metrics"]["exp_weighted_std"][i]
            wmean_exp_error = (
                results["metrics"]["exp_weighted_mean_error"][i] * 100
            )
            map_exp = results["metrics"]["exp_maximum_a_posteriori"][i]
            map_exp_error = (
                results["metrics"]["exp_maximum_a_posteriori_error"][i] * 100
            )

            distplot._fig.add_annotation(
                text=(
                    f"ground truth [&#952;]: {gt:.2f}<br>"
                    f"mean &#177; std [&#952;]: {wmean:.2f} &#177; "
                    f"{std:.2f}<br>"
                    f"map [&#952;]: {map:.2f}<br>"
                    f"mean error [%]: {wmean_error:.2f}<br>"
                    f"map error [%]: {map_error:.2f}<br>"
                    f"bandwidth [&#952;]: {bandwidth:.2f}<br>"
                    f"exp. ground truth [&#952;]: {gt_exp:.2f}<br>"
                    f"exp. mean &#177; std [&#952;]: {wmean_exp:.2f} &#177; "
                    f"{std_exp:.2f}<br>"
                    f"exp. map [&#952;]: {map_exp:.2f}<br>"
                    f"exp. mean error [%]: {wmean_exp_error:.2f}<br>"
                    f"exp. map error [%]: {map_exp_error:.2f}"
                ),
                align="left",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.0,
                bordercolor="white",
                xanchor="right",
                yanchor="top",
                borderpad=7,
            )
            report.add([distplot])

            # pressure_plot = visualizer.Plot2D(
            #     title="Pressure",
            #     xaxis_title=r"$s$",
            #     yaxis_title=r"$mmHg$",
            # )
            # bc_result = result_map[result_map.name == bc_map[bc_name]["name"]]
            # times = np.array(bc_result["time"])[-num_pts_per_cycle:]
            # times -= times[0]
            # pressure_plot.add_line_trace(
            #     x=times,
            #     y=taskutils.cgs_pressure_to_mmgh(
            #         bc_result[bc_map[bc_name]["pressure"]].iloc[
            #             -num_pts_per_cycle:
            #         ]
            #     ),
            #     **map_opts,
            # )
            # bc_result = result_mean[result_map.name == bc_map[bc_name]["name"]]
            # pressure_plot.add_line_trace(
            #     x=times,
            #     y=taskutils.cgs_pressure_to_mmgh(
            #         bc_result[bc_map[bc_name]["pressure"]].iloc[
            #             -num_pts_per_cycle:
            #         ]
            #     ),
            #     **mean_opts,
            # )

            # flow_plot = visualizer.Plot2D(
            #     title="Flow",
            #     xaxis_title=r"$s$",
            #     yaxis_title=r"$\frac{l}{min}$",
            # )
            # bc_result = result_map[result_map.name == bc_map[bc_name]["name"]]
            # flow_plot.add_line_trace(
            #     x=times,
            #     y=taskutils.cgs_flow_to_lmin(
            #         bc_result[bc_map[bc_name]["flow"]].iloc[
            #             -num_pts_per_cycle:
            #         ]
            #     ),
            #     **map_opts,
            # )
            # bc_result = result_mean[result_map.name == bc_map[bc_name]["name"]]
            # flow_plot.add_line_trace(
            #     x=times,
            #     y=taskutils.cgs_flow_to_lmin(
            #         bc_result[bc_map[bc_name]["flow"]].iloc[
            #             -num_pts_per_cycle:
            #         ]
            #     ),
            #     **mean_opts,
            # )

            # report.add([pressure_plot, flow_plot])

        return report

    def _get_raw_results(
        self, frame: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return raw queens result.

        Args:
            frame: Specify smc iteration to read results from. If None, the
                final result will be returned.

        Returns:
            particles: Coordinates of the particles.
            weights: Weights of particles.
            log_post: Log posterior of particles.
        """

        filename = (
            "results.pickle" if frame is None else f"results{frame}.pickle"
        )
        with open(
            os.path.join(self.output_folder, filename),
            "rb",
        ) as ff:
            results = pickle.load(ff)

        particles = np.array(results["raw_output_data"]["particles"])
        weights = np.array(results["raw_output_data"]["weights"])
        log_post = np.array(results["raw_output_data"]["log_posterior"])

        return particles, weights.flatten(), log_post


class _Forward_Model:
    """Windkessel tuning forward model.

    This forward model performs evaluations of a 0D model based on a
    given total resistance.
    """

    def __init__(self, zerod_config: reader.SvZeroDSolverInputHandler) -> None:
        """Construct the forward model.

        Args:
            zerod_config: 0D simulation config handler.
        """

        self.based_zerod = zerod_config
        self.base_config = zerod_config.data.copy()
        self.outlet_bcs = zerod_config.outlet_boundary_conditions
        self.outlet_bc_ids = []
        for i, bc in enumerate(zerod_config.data["boundary_conditions"]):
            if bc["bc_name"] in self.outlet_bcs:
                self.outlet_bc_ids.append(i)
        # self.base_config["simulation_parameters"].update(
        #     {"output_last_cycle_only": True, "output_interval": 10}
        # )

        bc_node_names = zerod_config.get_bc_node_names()
        self.inlet_dof_name = [
            f"pressure:{n}" for n in bc_node_names if "INFLOW" in n
        ][0]
        self.outlet_dof_names = [
            f"flow:{n}" for n in bc_node_names if "INFLOW" not in n
        ]

        # Distal to proximal resistance ratio at each outlet
        self._distal_to_proximal = [
            bc["bc_values"]["Rd"] / bc["bc_values"]["Rp"]
            for bc in self.outlet_bcs.values()
        ]

        # Time constants for each outlet
        self._time_constants = [
            bc["bc_values"]["Rd"] * bc["bc_values"]["C"]
            for bc in self.outlet_bcs.values()
        ]

        # Ratio to total values at each outlet
        self._total_ratio = {}
        for val in ["C", "Rp", "Rd"]:
            total = 0.0
            for bc in self.outlet_bcs.values():
                total += bc["bc_values"][val]
            self._total_ratio[val] = [
                bc["bc_values"][val] / total
                for bc in self.outlet_bcs.values()
            ]

    def to_file(self, filename: str):
        """Write configuration to 0D input file"""
        self.based_zerod.to_file(filename)

    def simulate_csv(self, filename: str):
        """Run forward simulation with base configuration and save results to csv"""
        svzerodplus.simulate(self.base_config).to_csv(filename)

    def simulate(self, sample: np.ndarray) -> Solver:
        """Run forward simulation with sample and return the solver object"""
        config = self.base_config.copy()

        # Change boundary conditions (set in derived class)
        self.change_boundary_conditions(config["boundary_conditions"], sample)

        # Run simulation
        try:
            solver = Solver(config)
            solver.run()
            return solver
        except RuntimeError:
            return None

    def evaluate(self, sample: np.ndarray) -> np.ndarray:
        """Objective function for the optimization"""
        raise NotImplementedError

    def change_boundary_conditions(self, boundary_conditions, sample):
        """Specify how boundary conditions are set with parameters"""
        raise NotImplementedError


class _Forward_ModelRpRd(_Forward_Model):
    def change_boundary_conditions(self, boundary_conditions, sample):
        # Set new total resistance at each outlet
        for i, bc_id in enumerate(self.outlet_bc_ids):
            ki = np.exp(sample[i])
            bc_values = boundary_conditions[bc_id]["bc_values"]
            bc_values["Rp"] = ki / (1.0 + self._distal_to_proximal[i])
            bc_values["Rd"] = ki - bc_values["Rp"]
            bc_values["C"] = self._time_constants[i] / bc_values["Rd"]

    def evaluate(self, sample: np.ndarray) -> np.ndarray:
        """Evaluates the sum of the offsets for the input output pressure relation
        for each outlet."""
        solver = self.simulate(sample)
        if solver is None:
            return np.array([9e99] * (len(self.outlet_dof_names) + 2))

        # Extract minimum and maximum inlet pressure for last cardiac cycle
        p_inlet = solver.get_single_result(self.inlet_dof_name)

        # Extract mean outlet pressure for last cardiac cycle at each BC
        q_outlet_mean = [
            solver.get_single_result_avg(dof) for dof in self.outlet_dof_names
        ]

        return np.array([p_inlet.min(), p_inlet.max(), *q_outlet_mean])


class _Forward_ModelRC(_Forward_Model):
    def change_boundary_conditions(self, boundary_conditions, sample):
        out_ids = range(len(self.outlet_bc_ids))
        # out_ids = [4]
        for i, val in enumerate(["Rd", "C"]):
            for j in out_ids:
                bc_values = boundary_conditions[self.outlet_bc_ids[j]]["bc_values"]
                bc_values[val] = np.exp(sample[i]) * self._total_ratio[val][j]
        # # only modify one boundary condition
        # if len(self.outlet_bc_ids) == 1:
        #     out_id = -1
        # else:
        #     out_id = 1
        # bc_id = self.outlet_bc_ids[out_id]
        # bc_values = boundary_conditions[bc_id]["bc_values"]
        
        # select variation
        # self.vary_r_ratio_c_ratio(bc_values, sample)
        # self.vary_rp_c(bc_values, sample)
        # self.vary_rd_c_0104_0001(bc_values, sample)
        # self.vary_rp_c_0104_0001(bc_values, sample)

        # r_ratio = 3163.0 / 256.0
        # bc_values["Rd"] = np.exp(sample[0])
        # bc_values["Rp"] = bc_values["Rd"] * r_ratio

        # c_total = np.exp(sample[1])
        # bc_values = boundary_conditions[bc_id]["bc_values"]
        # for i, bc_id in enumerate(self.outlet_bc_ids):
        #     bc_values = boundary_conditions[bc_id]["bc_values"]
        #     bc_values["C"] = self._total_ratio["C"][i] * c_total

    def vary_r_ratio_c_ratio(self, bc_values, sample):
        # variable Rd / Rp
        r_ratio = np.exp(sample[0])

        # variable Rd * C
        rc_ratio = np.exp(sample[1])

        # const Rp
        bc_values["Rd"] = r_ratio * bc_values["Rp"]
        bc_values["C"] = rc_ratio / bc_values["Rd"]

    def vary_rp_c(self, bc_values, sample):
        # const Rp / Rd
        r_ratio = 100.0
        r0 = np.exp(sample[0]) / (r_ratio + 1.0)

        # variable Rp, C
        bc_values["Rp"] = r0
        bc_values["Rd"] = r0 * r_ratio
        bc_values["C"] = np.exp(sample[1])

    # def vary_rd_c_0104_0001(self, bc_values, sample):
    #     # for i, bc_id in enumerate(self.outlet_bc_ids):
    #     #     bc_values = boundary_conditions[bc_id]["bc_values"]
    #     #     bc_values["C"] = self._total_ratio["C"][i] * c_total
    #     # from ground truth
    #     r_ratio = 256.0 / 3163.0

    #     # variable Rp, C
    #     bc_values["Rd"] = np.exp(sample[0])
    #     bc_values["Rp"] = bc_values["Rd"] * r_ratio
    #     bc_values["C"] = np.exp(sample[1])

    # def vary_rp_c_0104_0001(self, bc_values, sample):
    #     # from ground truth
    #     r_ratio = 3163.0 / 256.0

    #     # variable Rp, C
    #     bc_values["Rp"] = np.exp(sample[0])
    #     bc_values["Rd"] = bc_values["Rp"] * r_ratio
    #     bc_values["C"] = np.exp(sample[1])

    def evaluate(self, sample: np.ndarray) -> np.ndarray:
        """Get the pressure curve at the inlet"""
        solver = self.simulate(sample)
        if solver is None:
            nt = self.base_config["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"]
            return np.array([9e99] * nt)
        
        p_inlet = solver.get_single_result(self.inlet_dof_name)
        # q_outlet = solver.get_single_result(self.outlet_dof_names[1])
        
        # tmax = self.base_config["boundary_conditions"][0]["bc_values"]["t"][-1]
        # dt = tmax / nt
        # dp_inlet = np.gradient(p_inlet, dt)

        return p_inlet
        # return np.array([p_inlet.max(), p_inlet.min()])
        # return np.array([p_inlet.max(), p_inlet.min(), dp_inlet.max()])
        # return np.array([p_inlet.max(), p_inlet.min(), np.mean(q_outlet)])


class _SMCRunner:
    def __init__(
        self,
        forward_model: _Forward_Model,
        y_obs: np.ndarray,
        len_theta: int,
        likelihood_std_vector: np.ndarray,
        prior_bounds: np.ndarray,
        num_particles: int,
        resampling_strategy: str,
        resampling_threshold: float,
        num_rejuvenation_steps: int,
        num_procs: int,
        console: Any,
    ):
        likelihood = stats.multivariate_normal(mean=np.zeros(len(y_obs)))

        prior = dists.StructDist(
            {
                f"k{i}": dists.Uniform(a=prior_bounds[i][0], b=prior_bounds[i][1])
                for i in range(len_theta)
            }
        )
        self.console = console
        self.len_theta = len_theta

        class StaticModel(ssp.StaticModel):
            def __init__(
                self, prior: dists.StructDist, len_theta: int
            ) -> None:
                super().__init__(None, prior)
                self.len_theta = len_theta

            def loglik(
                self, theta: np.ndarray, t: Optional[int] = None
            ) -> np.ndarray:
                results = []
                with get_context("fork").Pool(num_procs) as pool:
                    with Progress(
                        " " * 20 + "Evaluating samples... ",
                        BarColumn(),
                        "{task.completed}/{task.total} completed | "
                        "{task.speed} samples/s",
                        console=console,
                    ) as progress:
                        for res in progress.track(
                            pool.imap(forward_model.evaluate, theta, 1),
                            total=len(theta),
                        ):
                            results.append(res)
                return likelihood.logpdf(
                    (np.array(results) - y_obs) / likelihood_std_vector
                )

        static_model = StaticModel(prior, self.len_theta)

        fk_model = ssp.AdaptiveTempering(
            model=static_model, len_chain=1 + num_rejuvenation_steps
        )

        self.runner = particles.SMC(
            fk=fk_model,
            N=num_particles,
            resampling=resampling_strategy,
            ESSrmin=resampling_threshold,
            verbose=False,
        )

    def run(self) -> tuple[list, list, list]:
        all_particles = []
        all_weights = []
        all_logpost = []

        for _ in self.runner:
            particles = np.array(
                [
                    self.runner.X.theta[f"k{i}"].flatten()
                    for i in range(self.len_theta)
                ]
            ).T.tolist()
            weights = np.array(self.runner.W).tolist()
            logpost = np.array(self.runner.X.lpost).tolist()
            self.console.log(
                f"Completed SMC step {self.runner.t} | "
                f"[yellow]ESS[/yellow]: {self.runner.wgts.ESS:.2f} | "
                "[yellow]tempering exponent[/yellow]: "
                f"{self.runner.X.shared['exponents'][-1]:.2e}"
            )
            all_particles.append(particles)
            all_weights.append(weights)
            all_logpost.append(logpost)

        return all_particles, all_weights, all_logpost
    
def joint_plot(x, y, weights, output_path):
    g = sns.JointGrid()

    cmap = plt.get_cmap('BuGn')
    color = np.array(cmap(1000))[:-1]

    # Create a hexbin plot with weights
    n_grid = 50
    g.ax_joint.hexbin(x, y, C=weights, gridsize=n_grid, linewidths=0.5, edgecolors='0.3', reduce_C_function=np.sum, cmap='BuGn')
    g.ax_marg_x.hist(x=x, weights=weights, bins=n_grid, color=color, alpha=0.5)
    g.ax_marg_y.hist(x=y, weights=weights, bins=n_grid, orientation="horizontal", color=color, alpha=0.5)

    g._figure.savefig(output_path, dpi=400)
    # g._figure.close()

def joint_plot2(x, y, weights, lims, output_path):
    results = {"theta_1": x, "theta_2": y}
    data = pd.DataFrame(data=results)

    joint = sns.jointplot(data=data, x="theta_1", y="theta_2", kind="hist", weights=weights, bins=100)
    joint.ax_joint.set_xlim(lims[0])
    joint.ax_joint.set_ylim(lims[1])
    # joint.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
    # joint.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
    joint._figure.savefig(output_path, bbox_inches='tight')

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child.

    Call with ForkedPdb().set_trace()
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin