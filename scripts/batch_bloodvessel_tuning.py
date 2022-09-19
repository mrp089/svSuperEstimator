import os
from svsuperestimator.main import run_from_config
import click
from rich import print


@click.command()
@click.argument("model_folder")
@click.argument("centerline_folder")
@click.option("--num_procs", default=1, help="Number of processors to use.")
@click.option(
    "--override",
    is_flag=True,
    default=False,
)
def main(model_folder, centerline_folder, num_procs, override):
    for model in [
        n for n in os.listdir(model_folder) if not n.startswith(".")
    ]:
        if not override and os.path.exists(
            os.path.join(
                model_folder,
                model,
                "ParameterEstimation",
                "BloodVesselTuning",
                "report.html",
            )
        ):
            print(
                f"Optimization for model {model} [bold orange]exists[/bold orange]"
            )
            continue

        config = {
            "project": os.path.join(model_folder, model),
            "global": {"num_procs": num_procs},
            "tasks": {
                "BloodVesselTuning": {
                    "threed_solution_file": os.path.join(
                        centerline_folder, f"{model}.vtp"
                    ),
                    "maxfev": 2000,
                }
            },
        }
        try:
            run_from_config(config)
            print(
                f"Optimization for model {model} [bold green]successful[/bold green]"
            )
        except Exception as excp:
            print(
                f"Optimization for model {model} [bold red]failed[/bold red] with {excp}"
            )


if __name__ == "__main__":
    main()
