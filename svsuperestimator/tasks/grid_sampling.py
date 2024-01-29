"""This module holds the GridSampling task."""
from __future__ import annotations

import pdb
import os
import pickle
from datetime import datetime
from multiprocessing import get_context
from typing import Any, Dict, Optional

import numpy as np
import orjson
import pandas as pd
import particles
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
from .windkessel_tuning import WindkesselTuning, _Forward_Model, _Forward_ModelRC, _Forward_ModelRpRd, _Forward_ModelRpRd_aorta, joint_plot, joint_plot2


class GridSampling(WindkesselTuning):
    """GridSamlping task"""

    TASKNAME = "grid_sampling"

    # todo: replace defaults

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
        elif model == "aorta":
            self.forward_model = _Forward_ModelRpRd_aorta(zerod_config_handler)
        else:
            raise NotImplementedError("Unknown forward model " + model)

        # Determine target observations through one forward evaluation
        y_obs = np.array(self.config["y_obs"])
        if y_obs.size == 0:
            y_obs = self.forward_model.evaluate(None)
        self.log("Setting target observation to:", y_obs)
        self.database["y_obs"] = y_obs.tolist()

        # Determine noise covariance
        std_vector = self.config["noise_factor"] * y_obs
        self.log("Setting std vector to:", std_vector)
        self.database["y_obs_std"] = std_vector.tolist()

        # Setup the iterator
        self.log("Setup tuning process")
        smc_runner = _GridRunner(
            forward_model=self.forward_model,
            y_obs=y_obs,
            len_theta=len(theta_obs),
            likelihood_std_vector=std_vector,
            prior_bounds=self.config["theta_range"],
            num_procs=self.config["num_procs"],
            num_samples=int(np.sqrt(self.config["num_particles"])),
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

    def generate_report(self) -> visualizer.Report:
        particles = np.array(self.database["particles"][-1])
        weights = np.array(self.database["weights"][-1])

        # Create distribition plots for all boundary conditions
        n_dim = particles.shape[1]
        if n_dim == 2:
            joint_plot(particles[:, 0], particles[:, 1], weights, self.config["theta_range"], "bivariate.png")

class _GridRunner:
    def __init__(
        self,
        forward_model: _Forward_Model,
        y_obs: np.ndarray,
        len_theta: int,
        likelihood_std_vector: np.ndarray,
        prior_bounds: tuple,
        num_samples: int,
        num_procs: int,
        console: Any,
    ):
        # print(prior_bounds)
        self.likelihood = stats.multivariate_normal(mean=np.zeros(len(y_obs)))
        self.y_obs = y_obs
        self.forward_model = forward_model
        self.likelihood_std_vector = likelihood_std_vector
        self.prior_bounds = prior_bounds
        self.num_samples = num_samples

        self.console = console
        self.len_theta = len_theta
        self.num_procs = num_procs

    def loglik(self, theta: np.ndarray, t: Optional[int] = None) -> np.ndarray:
        results = []
        with get_context("fork").Pool(self.num_procs) as pool:
            with Progress(
                " " * 20 + "Evaluating samples... ",
                BarColumn(),
                "{task.completed}/{task.total} completed | "
                "{task.speed} samples/s",
                console=self.console,
            ) as progress:
                for res in progress.track(
                    pool.imap(self.forward_model.evaluate, theta, 1),
                    total=len(theta),
                ):
                    results.append(res)
        return self.likelihood.logpdf(
            (np.array(results) - self.y_obs) / self.likelihood_std_vector
        )

    def run(self) -> tuple[list, list, list]:
        ranges = [
            np.linspace(
                self.prior_bounds[i][0], self.prior_bounds[i][1], self.num_samples
            )
            for i in range(self.len_theta)
        ]

        all_particles = np.array(np.meshgrid(*ranges)).T.reshape(
            -1, self.len_theta
        )

        all_logpost = self.loglik(all_particles)
        all_weights = np.exp(all_logpost)
        all_weights /= np.sum(all_weights)

        return [all_particles], [all_weights], [all_logpost]
