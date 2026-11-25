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


def _add_heffte_exp(
    workspace_name,
    workload,
    exp_name="test_exp",
    variants=None,
    extra_vars=None,
):
    global_args = ["-w", workspace_name]
    args = [
        "manage",
        "experiments",
        "heffte",
        "--wf",
        workload,
        "-e",
        exp_name,
        "-v",
        "n_nodes=1",
        "-v",
        "processes_per_node=1",
        "-v",
        "mpi_command=mpirun -n {n_ranks}",
        "-v",
        "batch_submit={execute_experiment}",
        "-v",
        "heffte_path=/opt/heffte",
    ]
    for var_spec in extra_vars or []:
        args.extend(["-v", var_spec])
    for var_spec in variants or []:
        args.extend(["-V", var_spec])
    workspace(*args, global_args=global_args)


def test_heffte_cuda_defaults_fft_and_mps(workspace_name):
    """Verify +cuda defaults fft to cufft and mps to -mps only on 3d_workloads."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _add_heffte_exp(
            workspace_name, "c2c", exp_name="cpu_c2c", variants=["cuda=false"]
        )
        _add_heffte_exp(
            workspace_name, "c2c", exp_name="gpu_c2c", variants=["cuda=true"]
        )
        _add_heffte_exp(
            workspace_name, "r2c", exp_name="gpu_r2c", variants=["cuda=true"]
        )
        _add_heffte_exp(
            workspace_name, "r2r", exp_name="gpu_r2r", variants=["cuda=true"]
        )
        _add_heffte_exp(
            workspace_name,
            "convolution",
            exp_name="gpu_conv",
            variants=["cuda=true"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        def read_script(wl, exp):
            path = os.path.join(
                ws.experiment_dir, "heffte", wl, exp, "execute_experiment"
            )
            with open(path, encoding="utf-8") as f:
                return f.read()

        cpu_c2c = read_script("c2c", "cpu_c2c")
        assert "speed3d_c2c stock " in cpu_c2c
        assert "-mps" not in cpu_c2c

        gpu_c2c = read_script("c2c", "gpu_c2c")
        assert "speed3d_c2c cufft " in gpu_c2c
        assert "-mps" in gpu_c2c

        gpu_r2c = read_script("r2c", "gpu_r2c")
        assert "speed3d_r2c cufft " in gpu_r2c
        assert "-mps" in gpu_r2c

        gpu_r2r = read_script("r2r", "gpu_r2r")
        assert "speed3d_r2r cufft-cos " in gpu_r2r
        assert "-mps" in gpu_r2r

        # convolution only supports CPU backends; must remain stock without -mps
        gpu_conv = read_script("convolution", "gpu_conv")
        assert "benchmarks/convolution stock " in gpu_conv
        assert "-mps" not in gpu_conv


def test_heffte_spack_gcc14_and_nvhpc_concretize(workspace_name):
    """Verify spack concretize wires gcc14/intel-mpi for ~nvhpc and nvhpc/nvhpc-mpi for +nvhpc."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        workspace(
            "manage",
            "experiments",
            "heffte",
            "--wf",
            "c2c",
            "-e",
            "spack_cpu",
            "-V",
            "package_manager=spack",
            "-v",
            "n_nodes=1",
            "-v",
            "processes_per_node=1",
            "-v",
            "mpi_command=mpirun -n {n_ranks}",
            "-v",
            "batch_submit={execute_experiment}",
            "-v",
            "env_name=env_heffte_gcc",
            global_args=global_args,
        )
        workspace(
            "manage",
            "experiments",
            "heffte",
            "--wf",
            "c2c",
            "-e",
            "spack_gpu_nvhpc",
            "-V",
            "package_manager=spack",
            "-V",
            "cuda=true",
            "-V",
            "nvhpc=true",
            "-v",
            "n_nodes=1",
            "-v",
            "processes_per_node=1",
            "-v",
            "mpi_command=mpirun -n {n_ranks}",
            "-v",
            "batch_submit={execute_experiment}",
            "-v",
            "env_name=env_heffte_nvhpc",
            global_args=global_args,
        )
        workspace("concretize", global_args=global_args)
        with open(ws.config_file_path, encoding="utf-8") as f:
            cfg = f.read()
        assert "openmpi" not in cfg

        parsed = syaml.load(cfg)
        pkgs = parsed["ramble"]["software"]["packages"]
        envs = parsed["ramble"]["software"]["environments"]

        assert pkgs["gcc14"]["pkg_spec"] == "gcc@14.2.0"
        assert pkgs["nvhpc"]["pkg_spec"] == "nvhpc +mpi"
        assert pkgs["nvhpc"]["compiler_spec"] == "nvhpc"
        assert pkgs["nvhpc-mpi"]["pkg_spec"] == "nvhpc +mpi"
        assert (
            pkgs["heffte-{application::heffte::version}"]["compiler"]
            == "gcc14"
        )
        assert (
            pkgs["heffte-{application::heffte::version}-nvhpc"]["compiler"]
            == "nvhpc"
        )

        assert "intel-mpi" in envs["env_heffte_gcc"]["packages"]
        assert "nvhpc-mpi" not in envs["env_heffte_gcc"]["packages"]
        assert (
            "heffte-{application::heffte::version}"
            in envs["env_heffte_gcc"]["packages"]
        )

        assert "nvhpc-mpi" in envs["env_heffte_nvhpc"]["packages"]
        assert "intel-mpi" not in envs["env_heffte_nvhpc"]["packages"]
        assert (
            "heffte-{application::heffte::version}-nvhpc"
            in envs["env_heffte_nvhpc"]["packages"]
        )


def test_heffte_nvhpc_requires_cuda_conflict(workspace_name):
    """Verify +nvhpc without +cuda fails validation."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name):
        _add_heffte_exp(
            workspace_name,
            "c2c",
            variants=["nvhpc=true", "cuda=false"],
        )
        with pytest.raises(ramble.error.ObjectValidationError):
            workspace("setup", "--dry-run", global_args=global_args)


def test_heffte_tolerance_and_max_error_foms(workspace_name):
    """Verify fixed Tolerance / Maximum error regexes match scientific notation."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _add_heffte_exp(workspace_name, "c2c", exp_name="ok_exp")
        _add_heffte_exp(workspace_name, "c2c", exp_name="bad_exp")
        workspace("setup", "--dry-run", global_args=global_args)

        ok_log = os.path.join(
            ws.experiment_dir, "heffte", "c2c", "ok_exp", "ok_exp.out"
        )
        with open(ok_log, "w", encoding="utf-8") as f:
            f.write(
                "Time per run: 0.012345 (s)\n"
                "Performance: 456.789 GFlops/s\n"
                "Memory usage: 128.5MB/rank\n"
                "Tolerance: 1.0e-05\n"
                "Max error: 3.45E-07\n"
            )

        bad_log = os.path.join(
            ws.experiment_dir, "heffte", "c2c", "bad_exp", "bad_exp.out"
        )
        with open(bad_log, "w", encoding="utf-8") as f:
            f.write(
                "Time per run: 0.012345 (s)\n"
                "Tolerance: 1e-05\n"
                "Max error: 2.5e-02\n"
                "ERROR: observed error after heFFTe benchmark exceeds the tolerance\n"
            )

        workspace("analyze", "-f", "json", global_args=global_args)
        result_file = os.path.join(ws.results_dir, "results.latest.json")
        with open(result_file, encoding="utf-8") as f:
            results = json.load(f)

        by_name = {exp["name"]: exp for exp in results["experiments"]}
        ok_res = by_name["heffte.c2c.ok_exp"]
        assert ok_res["RAMBLE_STATUS"] == "SUCCESS"
        null_foms = {
            f["name"]: f
            for ctx in ok_res["CONTEXTS"]
            if ctx["name"] == "null"
            for f in ctx["foms"]
        }
        assert null_foms["Time per run"]["value"] == "0.012345"
        assert null_foms["Performance"]["value"] == "456.789"
        assert null_foms["Memory usage per rank"]["value"] == "128.5"
        assert null_foms["Tolerance"]["value"] == "1.0e-05"
        assert null_foms["Maximum error"]["value"] == "3.45E-07"

        bad_res = by_name["heffte.c2c.bad_exp"]
        assert bad_res["RAMBLE_STATUS"] == "FAILED"
