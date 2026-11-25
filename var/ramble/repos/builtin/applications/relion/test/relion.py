# Copyright 2022-2026 The Ramble Authors
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import json
import os

import pytest

import ramble.error
import ramble.workspace
from ramble.main import RambleCommand

import spack.util.spack_yaml as syaml

workspace = RambleCommand("workspace")

pytestmark = pytest.mark.usefixtures(
    "mutable_config",
    "mutable_mock_workspace_path",
)


def _setup_relion_exp(
    workspace_name,
    workload,
    exp_name="test_exp",
    extra_vars=None,
    variants=None,
):
    global_args = ["-w", workspace_name]
    args = [
        "manage",
        "experiments",
        "relion",
        "--wf",
        workload,
        "-e",
        exp_name,
        "-v",
        "n_nodes=1",
        "-v",
        "mpi_command=mpirun -n {n_ranks}",
        "-v",
        "batch_submit={execute_experiment}",
    ]
    for var_spec in extra_vars or []:
        args.extend(["-v", var_spec])
    for var_spec in variants or []:
        args.extend(["-V", var_spec])
    workspace(*args, global_args=global_args)


def test_relion_prepare_analysis_units_and_elapsed_time(workspace_name):
    """Verify _prepare_analysis parses elapsed (not projected total) across sec/min/hrs."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _setup_relion_exp(
            workspace_name,
            "class2d",
            extra_vars=["processes_per_node=2", "n_iter=3"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        run_dir = os.path.join(
            ws.experiment_dir, "relion", "class2d", "test_exp"
        )
        log_file = os.path.join(run_dir, "test_exp.out")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(
                "RELION version: 5.0.1 (release-build)\n"
                "Precision: BASE=double, VECTOR-ACC=single\n"
                " Expectation iteration 1 of 3\n"
                '  40/  35 sec ............................................................~~(,_,"> yum!\n'
                " Expectation iteration 2 of 3\n"
                '22.63/22.55 min ............................................................~~(,_,"> yum!\n'
                " Expectation iteration 3 of 3\n"
                ' 1.50/ 1.25 hrs ............................................................~~(,_,"> yum!\n'
            )

        model_file = os.path.join(run_dir, "run_it003_model.star")
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("_rlnCurrentResolution                       7.537472\n")

        workspace("analyze", "-f", "json", global_args=global_args)

        result_file = os.path.join(ws.results_dir, "results.latest.json")
        with open(result_file, encoding="utf-8") as f:
            results = json.load(f)

        exp_res = results["experiments"][0]
        assert exp_res["RAMBLE_STATUS"] == "SUCCESS"

        contexts = {ctx["name"]: ctx for ctx in exp_res["CONTEXTS"]}
        null_foms = {f["name"]: f for f in contexts["null"]["foms"]}
        assert null_foms["RELION version"]["value"] == "5.0.1"
        assert (
            null_foms["Precision"]["value"] == "BASE=double, VECTOR-ACC=single"
        )
        assert int(null_foms["Completed iterations"]["value"]) == 3
        assert float(null_foms["Final resolution"]["value"]) == pytest.approx(
            7.537472
        )

        # Verify elapsed values (40 sec, 22.63 min = 1357.8 s, 1.50 hrs = 5400 s)
        # rather than projected totals (35, 1353.0, 4500.0).
        expected_e_step = {
            "Iteration 1": 40.0,
            "Iteration 2": 22.63 * 60.0,
            "Iteration 3": 1.50 * 3600.0,
        }
        for ctx_name, expected in expected_e_step.items():
            foms = {f["name"]: f for f in contexts[ctx_name]["foms"]}
            assert float(
                foms["Iteration E-step time"]["value"]
            ) == pytest.approx(expected)
            assert "Iteration wall time" not in foms

        assert float(null_foms["Total E-step time"]["value"]) == pytest.approx(
            sum(expected_e_step.values())
        )
        assert "Total iteration wall time" not in null_foms


def test_relion_prepare_analysis_checkpoint_mtimes(workspace_name):
    """Verify wall times use checkpoint mtimes when distinct and skip zero deltas."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _setup_relion_exp(
            workspace_name,
            "class3d",
            extra_vars=["processes_per_node=2", "n_iter=2"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        run_dir = os.path.join(
            ws.experiment_dir, "relion", "class3d", "test_exp"
        )
        log_file = os.path.join(run_dir, "test_exp.out")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(
                " Expectation iteration 1 of 2\n"
                '  10/  10 sec ............................................................~~(,_,"> yum!\n'
                " Expectation iteration 2 of 2\n"
                '  20/  20 sec ............................................................~~(,_,"> yum!\n'
            )

        model_file = os.path.join(run_dir, "run_it002_model.star")
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("_rlnCurrentResolution 6.25\n")

        # Distinct mtimes yield positive wall time deltas
        mtimes = {0: 1000.0, 1: 1015.5, 2: 1040.5}
        for it, ts in mtimes.items():
            star = os.path.join(run_dir, f"run_it{it:03d}_optimiser.star")
            with open(star, "w", encoding="utf-8") as f:
                f.write("data_\n")
            os.utime(star, (ts, ts))

        workspace("analyze", "-f", "json", global_args=global_args)
        result_file = os.path.join(ws.results_dir, "results.latest.json")
        with open(result_file, encoding="utf-8") as f:
            results = json.load(f)

        contexts = {
            ctx["name"]: ctx for ctx in results["experiments"][0]["CONTEXTS"]
        }
        null_foms = {f["name"]: f for f in contexts["null"]["foms"]}
        assert float(
            null_foms["Total iteration wall time"]["value"]
        ) == pytest.approx(40.5)
        it1_foms = {f["name"]: f for f in contexts["Iteration 1"]["foms"]}
        it2_foms = {f["name"]: f for f in contexts["Iteration 2"]["foms"]}
        assert float(
            it1_foms["Iteration wall time"]["value"]
        ) == pytest.approx(15.5)
        assert float(
            it2_foms["Iteration wall time"]["value"]
        ) == pytest.approx(25.0)

        # Identical mtimes (simulating copied/archived files) should not emit 0.0 s wall times
        for it in (0, 1, 2):
            star = os.path.join(run_dir, f"run_it{it:03d}_optimiser.star")
            os.utime(star, (2000.0, 2000.0))

        workspace("analyze", "-f", "json", global_args=global_args)
        with open(result_file, encoding="utf-8") as f:
            results = json.load(f)

        contexts = {
            ctx["name"]: ctx for ctx in results["experiments"][0]["CONTEXTS"]
        }
        null_foms = {f["name"]: f for f in contexts["null"]["foms"]}
        assert "Total iteration wall time" not in null_foms
        it1_foms = {f["name"]: f for f in contexts["Iteration 1"]["foms"]}
        assert "Iteration wall time" not in it1_foms


def test_relion_refine3d_ignores_higher_scoped_n_iter(workspace_name):
    """Verify refine3d checks run_model.star and enforces Completed Iterations even if workspace defines n_iter."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _setup_relion_exp(
            workspace_name,
            "refine3d",
            exp_name="passed_exp",
            extra_vars=["processes_per_node=3", "n_iter=25"],
        )
        _setup_relion_exp(
            workspace_name,
            "refine3d",
            exp_name="zero_iter_exp",
            extra_vars=["processes_per_node=3", "n_iter=25"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        # passed_exp: has 1 completed iteration and run_model.star (NOT run_it025_model.star)
        passed_dir = os.path.join(
            ws.experiment_dir, "relion", "refine3d", "passed_exp"
        )
        with open(
            os.path.join(passed_dir, "passed_exp.out"), "w", encoding="utf-8"
        ) as f:
            f.write(
                " Expectation iteration 1\n"
                '  12/  12 sec ............................................................~~(,_,"> yum!\n'
            )
        with open(
            os.path.join(passed_dir, "run_model.star"), "w", encoding="utf-8"
        ) as f:
            f.write("_rlnCurrentResolution 3.45\n")

        # zero_iter_exp: has run_model.star but 0 iterations completed
        zero_dir = os.path.join(
            ws.experiment_dir, "relion", "refine3d", "zero_iter_exp"
        )
        with open(
            os.path.join(zero_dir, "zero_iter_exp.out"), "w", encoding="utf-8"
        ) as f:
            f.write("RELION version: 5.0.1\n")
        with open(
            os.path.join(zero_dir, "run_model.star"), "w", encoding="utf-8"
        ) as f:
            f.write("_rlnCurrentResolution 3.45\n")

        workspace("analyze", "-f", "json", global_args=global_args)
        result_file = os.path.join(ws.results_dir, "results.latest.json")
        with open(result_file, encoding="utf-8") as f:
            results = json.load(f)

        by_name = {exp["name"]: exp for exp in results["experiments"]}
        passed_res = by_name["relion.refine3d.passed_exp"]
        assert passed_res["RAMBLE_STATUS"] == "SUCCESS"
        contexts = {ctx["name"]: ctx for ctx in passed_res["CONTEXTS"]}
        null_foms = {f["name"]: f for f in contexts["null"]["foms"]}
        assert float(null_foms["Final resolution"]["value"]) == pytest.approx(
            3.45
        )
        assert int(null_foms["Completed iterations"]["value"]) == 1

        zero_res = by_name["relion.refine3d.zero_iter_exp"]
        assert zero_res["RAMBLE_STATUS"] == "FAILED"


def test_relion_stale_stats_removed_on_rerun_failure(workspace_name):
    """Verify stale relion_stats.out is deleted when a re-run produces 0 iterations."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _setup_relion_exp(
            workspace_name,
            "class2d",
            extra_vars=["processes_per_node=2", "n_iter=1"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        run_dir = os.path.join(
            ws.experiment_dir, "relion", "class2d", "test_exp"
        )
        stats_path = os.path.join(run_dir, "relion_stats.out")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(
                "Iteration 1: e_step_time = 10.0000 s\n"
                "Completed iterations: 1\n"
                "Final resolution: 5.0 Angstroms\n"
            )
        # Re-run crashes before iteration 1 completes
        with open(
            os.path.join(run_dir, "test_exp.out"), "w", encoding="utf-8"
        ) as f:
            f.write(" Expectation iteration 1 of 1\nsegmentation fault\n")

        workspace("analyze", "-f", "json", global_args=global_args)
        assert not os.path.exists(stats_path)

        result_file = os.path.join(ws.results_dir, "results.latest.json")
        with open(result_file, encoding="utf-8") as f:
            results = json.load(f)
        assert results["experiments"][0]["RAMBLE_STATUS"] == "FAILED"


@pytest.mark.parametrize(
    "error_line",
    [
        "=== RELION_ERROR ===\n",
        "ERROR: MPI_Abort called\n",
    ],
)
def test_relion_fatal_error_anti_match(workspace_name, error_line):
    """Verify fatal error banners mark the experiment as FAILED."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _setup_relion_exp(
            workspace_name,
            "class2d",
            extra_vars=["processes_per_node=2", "n_iter=1"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        run_dir = os.path.join(
            ws.experiment_dir, "relion", "class2d", "test_exp"
        )
        with open(
            os.path.join(run_dir, "test_exp.out"), "w", encoding="utf-8"
        ) as f:
            f.write(
                " Expectation iteration 1 of 1\n"
                '  10/  10 sec ............................................................~~(,_,"> yum!\n'
                + error_line
            )
        with open(
            os.path.join(run_dir, "run_it001_model.star"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("_rlnCurrentResolution 4.0\n")

        workspace("analyze", "-f", "json", global_args=global_args)
        result_file = os.path.join(ws.results_dir, "results.latest.json")
        with open(result_file, encoding="utf-8") as f:
            results = json.load(f)
        assert results["experiments"][0]["RAMBLE_STATUS"] == "FAILED"


@pytest.mark.parametrize(
    "workload,ranks",
    [
        ("class2d", 1),
        ("class3d", 1),
        ("refine3d", 1),
        ("refine3d", 2),
        ("refine3d", 4),
    ],
)
def test_relion_mpi_rank_validators_reject_invalid(
    workspace_name, workload, ranks
):
    """Verify register_validator enforces minimum / odd MPI rank requirements."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name):
        _setup_relion_exp(
            workspace_name,
            workload,
            extra_vars=[f"processes_per_node={ranks}"],
        )
        with pytest.raises(ramble.error.ObjectValidationError):
            workspace("setup", "--dry-run", global_args=global_args)


def test_relion_accel_flags_and_conflicts(workspace_name):
    """Verify accel_flags defaults for +cuda / +altcpu and conflict checks."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _setup_relion_exp(
            workspace_name,
            "class2d",
            exp_name="gpu_exp",
            extra_vars=["processes_per_node=2"],
            variants=["cuda=true"],
        )
        _setup_relion_exp(
            workspace_name,
            "class2d",
            exp_name="altcpu_exp",
            extra_vars=["processes_per_node=2"],
            variants=["altcpu=true"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        gpu_script = os.path.join(
            ws.experiment_dir,
            "relion",
            "class2d",
            "gpu_exp",
            "execute_experiment",
        )
        with open(gpu_script, encoding="utf-8") as f:
            assert '--gpu ""' in f.read()

        altcpu_script = os.path.join(
            ws.experiment_dir,
            "relion",
            "class2d",
            "altcpu_exp",
            "execute_experiment",
        )
        with open(altcpu_script, encoding="utf-8") as f:
            assert "--cpu" in f.read()


@pytest.mark.parametrize(
    "variants",
    [
        ["cuda=true", "altcpu=true"],
        ["nvhpc=true", "cuda=false"],
    ],
)
def test_relion_variant_conflicts(workspace_name, variants):
    """Verify conflicting relion variants raise ObjectValidationError."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name):
        _setup_relion_exp(
            workspace_name,
            "class2d",
            extra_vars=["processes_per_node=2"],
            variants=variants,
        )
        with pytest.raises(ramble.error.ObjectValidationError):
            workspace("setup", "--dry-run", global_args=global_args)


def test_relion_refine3d_archive_includes_converged_optimiser(workspace_name):
    """Verify archive copies run_optimiser.star for refine3d."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _setup_relion_exp(
            workspace_name,
            "refine3d",
            extra_vars=["processes_per_node=3"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        run_dir = os.path.join(
            ws.experiment_dir, "relion", "refine3d", "test_exp"
        )
        optimiser_file = os.path.join(run_dir, "run_optimiser.star")
        with open(optimiser_file, "w", encoding="utf-8") as f:
            f.write("data_\n")

        workspace("archive", global_args=global_args)
        archived = os.path.join(
            ws.latest_archive_path,
            "experiments",
            "relion",
            "refine3d",
            "test_exp",
            "run_optimiser.star",
        )
        assert os.path.isfile(archived)


def test_relion_version_bin_variable(workspace_name):
    """Verify print-version uses relion_version_bin and allows custom overrides."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _setup_relion_exp(
            workspace_name,
            "class2d",
            exp_name="default_bin",
            extra_vars=["processes_per_node=2"],
        )
        _setup_relion_exp(
            workspace_name,
            "class2d",
            exp_name="custom_bin",
            extra_vars=[
                "processes_per_node=2",
                "relion_version_bin=/opt/relion/bin/relion_refine",
            ],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        default_script = os.path.join(
            ws.experiment_dir,
            "relion",
            "class2d",
            "default_bin",
            "execute_experiment",
        )
        with open(default_script, encoding="utf-8") as f:
            assert "relion_refine --version" in f.read()

        custom_script = os.path.join(
            ws.experiment_dir,
            "relion",
            "class2d",
            "custom_bin",
            "execute_experiment",
        )
        with open(custom_script, encoding="utf-8") as f:
            assert "/opt/relion/bin/relion_refine --version" in f.read()


def test_relion_spack_nvhpc_concretize(workspace_name):
    """Verify spack concretize wires gcc14/intel-mpi for ~nvhpc and nvhpc/nvhpc-mpi for +nvhpc while keeping wxwidgets on gcc14."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _setup_relion_exp(
            workspace_name,
            "class2d",
            exp_name="cpu_gcc",
            extra_vars=["processes_per_node=2", "env_name=env_relion_gcc"],
            variants=["package_manager=spack"],
        )
        _setup_relion_exp(
            workspace_name,
            "class2d",
            exp_name="gpu_nvhpc",
            extra_vars=["processes_per_node=2", "env_name=env_relion_nvhpc"],
            variants=["package_manager=spack", "cuda=true", "nvhpc=true"],
        )
        workspace("concretize", global_args=global_args)
        with open(ws.config_file_path, encoding="utf-8") as f:
            cfg = f.read()
        assert "openmpi" not in cfg
        assert "python" not in cfg

        parsed = syaml.load(cfg)
        pkgs = parsed["ramble"]["software"]["packages"]
        envs = parsed["ramble"]["software"]["environments"]

        assert pkgs["gcc14"]["pkg_spec"] == "gcc@14.2.0"
        assert pkgs["nvhpc"]["pkg_spec"] == "nvhpc +mpi"
        assert pkgs["nvhpc"]["compiler_spec"] == "nvhpc"
        assert pkgs["nvhpc-mpi"]["pkg_spec"] == "nvhpc +mpi"
        assert pkgs["wxwidgets"]["compiler"] == "gcc14"
        assert (
            pkgs["relion-{application::relion::version}"]["compiler"]
            == "gcc14"
        )
        assert (
            pkgs["relion-{application::relion::version}-nvhpc"]["compiler"]
            == "nvhpc"
        )

        assert "intel-mpi" in envs["env_relion_gcc"]["packages"]
        assert "nvhpc-mpi" not in envs["env_relion_gcc"]["packages"]
        assert "wxwidgets" in envs["env_relion_gcc"]["packages"]
        assert (
            "relion-{application::relion::version}"
            in envs["env_relion_gcc"]["packages"]
        )

        assert "nvhpc-mpi" in envs["env_relion_nvhpc"]["packages"]
        assert "intel-mpi" not in envs["env_relion_nvhpc"]["packages"]
        assert "wxwidgets" in envs["env_relion_nvhpc"]["packages"]
        assert (
            "relion-{application::relion::version}-nvhpc"
            in envs["env_relion_nvhpc"]["packages"]
        )


def test_relion_refine3d_converged_final_iteration_wall_time(workspace_name):
    """Verify refine3d maps run_optimiser.star mtime to the final converged iteration."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _setup_relion_exp(
            workspace_name,
            "refine3d",
            exp_name="converged_exp",
            extra_vars=["processes_per_node=3"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        run_dir = os.path.join(
            ws.experiment_dir, "relion", "refine3d", "converged_exp"
        )
        with open(
            os.path.join(run_dir, "converged_exp.out"), "w", encoding="utf-8"
        ) as f:
            f.write(
                " Expectation iteration 1\n"
                '   2/   2 sec ............................................................~~(,_,"> yum!\n'
                " Expectation iteration 2\n"
                '   7/   4 sec ............................................................~~(,_,"> yum!\n'
            )
        with open(
            os.path.join(run_dir, "run_model.star"), "w", encoding="utf-8"
        ) as f:
            f.write("_rlnCurrentResolution 16.634484\n")

        # Converged refine3d writes run_it000, run_it001, and run_optimiser.star (no run_it002)
        for fname, ts in (
            ("run_it000_optimiser.star", 1000.0),
            ("run_it001_optimiser.star", 1010.25),
            ("run_optimiser.star", 1090.0),
        ):
            star = os.path.join(run_dir, fname)
            with open(star, "w", encoding="utf-8") as f:
                f.write("data_\n")
            os.utime(star, (ts, ts))

        workspace("analyze", "-f", "json", global_args=global_args)
        result_file = os.path.join(ws.results_dir, "results.latest.json")
        with open(result_file, encoding="utf-8") as f:
            results = json.load(f)

        exp_res = results["experiments"][0]
        assert exp_res["RAMBLE_STATUS"] == "SUCCESS"
        contexts = {ctx["name"]: ctx for ctx in exp_res["CONTEXTS"]}
        null_foms = {f["name"]: f for f in contexts["null"]["foms"]}
        assert int(null_foms["Completed iterations"]["value"]) == 2
        assert float(
            null_foms["Total iteration wall time"]["value"]
        ) == pytest.approx(90.0)

        it1_foms = {f["name"]: f for f in contexts["Iteration 1"]["foms"]}
        it2_foms = {f["name"]: f for f in contexts["Iteration 2"]["foms"]}
        assert float(
            it1_foms["Iteration wall time"]["value"]
        ) == pytest.approx(10.25)
        assert float(
            it2_foms["Iteration wall time"]["value"]
        ) == pytest.approx(79.75)

        # If run_it002_optimiser.star also exists, do not invent a spurious Iteration 3
        star_it2 = os.path.join(run_dir, "run_it002_optimiser.star")
        with open(star_it2, "w", encoding="utf-8") as f:
            f.write("data_\n")
        os.utime(star_it2, (1050.0, 1050.0))

        workspace("analyze", "-f", "json", global_args=global_args)
        with open(result_file, encoding="utf-8") as f:
            results = json.load(f)
        contexts = {
            ctx["name"]: ctx for ctx in results["experiments"][0]["CONTEXTS"]
        }
        null_foms = {f["name"]: f for f in contexts["null"]["foms"]}
        assert int(null_foms["Completed iterations"]["value"]) == 2
        assert "Iteration 3" not in contexts
        assert float(
            null_foms["Total iteration wall time"]["value"]
        ) == pytest.approx(50.0)

        # Even if the log has no E-step iterations, do not invent Iteration 3 from run_optimiser.star
        with open(
            os.path.join(run_dir, "converged_exp.out"), "w", encoding="utf-8"
        ) as f:
            f.write("RELION version: 5.0.1\n")

        workspace("analyze", "-f", "json", global_args=global_args)
        with open(result_file, encoding="utf-8") as f:
            results = json.load(f)
        contexts = {
            ctx["name"]: ctx for ctx in results["experiments"][0]["CONTEXTS"]
        }
        assert "Iteration 3" not in contexts

        # RELION 3.x fallback: when run_optimiser.star is absent, use run_model.star mtime
        os.remove(star_it2)
        os.remove(os.path.join(run_dir, "run_optimiser.star"))
        os.utime(os.path.join(run_dir, "run_model.star"), (1075.0, 1075.0))
        with open(
            os.path.join(run_dir, "converged_exp.out"), "w", encoding="utf-8"
        ) as f:
            f.write(
                " Expectation iteration 1\n"
                '   2/   2 sec ............................................................~~(,_,"> yum!\n'
                " Expectation iteration 2\n"
                '   7/   4 sec ............................................................~~(,_,"> yum!\n'
            )

        workspace("analyze", "-f", "json", global_args=global_args)
        with open(result_file, encoding="utf-8") as f:
            results = json.load(f)
        contexts = {
            ctx["name"]: ctx for ctx in results["experiments"][0]["CONTEXTS"]
        }
        null_foms = {f["name"]: f for f in contexts["null"]["foms"]}
        assert float(
            null_foms["Total iteration wall time"]["value"]
        ) == pytest.approx(75.0)
        it2_foms = {f["name"]: f for f in contexts["Iteration 2"]["foms"]}
        assert float(
            it2_foms["Iteration wall time"]["value"]
        ) == pytest.approx(64.75)
