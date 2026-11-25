# Copyright 2022-2026 The Ramble Authors
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

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


def _add_gromacs_exp(
    workspace_name,
    workload,
    exp_name="test_exp",
    app_spec="gromacs",
    extra_vars=None,
    variants=None,
):
    global_args = ["-w", workspace_name]
    args = [
        "manage",
        "experiments",
        app_spec,
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


def _read_script(ws, app_name, workload, exp_name):
    script = os.path.join(
        ws.experiment_dir, app_name, workload, exp_name, "execute_experiment"
    )
    with open(script, encoding="utf-8") as f:
        return f.read()


def test_gromacs_cuda_pme_vs_rf(workspace_name):
    """Verify +cuda only passes -pme gpu / -npme for PME runs, not type=rf."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="pme_1",
            extra_vars=["processes_per_node=1", "type=pme"],
            variants=["cuda=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="pme_4",
            extra_vars=["processes_per_node=4", "type=pme"],
            variants=["cuda=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="rf_4",
            extra_vars=["processes_per_node=4", "type=rf"],
            variants=["cuda=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_bare",
            exp_name="bare_rf_4",
            extra_vars=["processes_per_node=4", "type=rf"],
            variants=["cuda=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_bare",
            exp_name="bare_pme_4",
            extra_vars=["processes_per_node=4", "type=pme"],
            variants=["cuda=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "lignocellulose",
            exp_name="ligno_4",
            extra_vars=["processes_per_node=4"],
            variants=["cuda=true"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        pme_1 = _read_script(ws, "gromacs", "water_gmx50", "pme_1")
        assert "mpirun -n 1 gmx_mpi mdrun" in pme_1
        assert "-nb gpu" in pme_1
        assert "-pme gpu" in pme_1
        assert "-npme" not in pme_1
        assert "-update gpu" in pme_1
        assert "GMX_GPU_DD_COMMS" not in pme_1
        assert "GMX_GPU_PME_PP_COMMS" not in pme_1
        assert "GMX_FORCE_UPDATE_DEFAULT_GPU" not in pme_1
        assert "GMX_ENABLE_DIRECT_GPU_COMM" not in pme_1

        pme_4 = _read_script(ws, "gromacs", "water_gmx50", "pme_4")
        assert "-pme gpu -npme 1" in pme_4

        rf_4 = _read_script(ws, "gromacs", "water_gmx50", "rf_4")
        assert "-nb gpu" in rf_4
        assert "-pme gpu" not in rf_4
        assert "-npme" not in rf_4

        bare_rf_4 = _read_script(ws, "gromacs", "water_bare", "bare_rf_4")
        assert "-pme gpu" not in bare_rf_4
        assert "-npme" not in bare_rf_4

        bare_pme_4 = _read_script(ws, "gromacs", "water_bare", "bare_pme_4")
        assert "-pme gpu -npme 1" in bare_pme_4

        ligno_4 = _read_script(ws, "gromacs", "lignocellulose", "ligno_4")
        assert "-nb gpu" in ligno_4
        assert "-pme gpu" not in ligno_4
        assert "-npme" not in ligno_4


def test_gromacs_thread_mpi_ntmpi_and_pme_offload(workspace_name):
    """Verify ~mpi omits mpirun, uses gmx with -ntmpi, and checks ntmpi for -npme."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _add_gromacs_exp(
            workspace_name,
            "stmv_pme",
            exp_name="single_rank_multi_ntmpi",
            extra_vars=["processes_per_node=1", "ntmpi=4"],
            variants=["mpi=false", "cuda=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "stmv_pme",
            exp_name="multi_rank_single_ntmpi",
            extra_vars=["processes_per_node=4", "ntmpi=1"],
            variants=["mpi=false", "cuda=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="water_pme_ntmpi4",
            extra_vars=["processes_per_node=1", "ntmpi=4", "type=pme"],
            variants=["mpi=false", "cuda=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="water_rf_ntmpi4",
            extra_vars=["processes_per_node=1", "ntmpi=4", "type=rf"],
            variants=["mpi=false", "cuda=true"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        s1 = _read_script(ws, "gromacs", "stmv_pme", "single_rank_multi_ntmpi")
        assert "mpirun" not in s1
        assert "gmx mdrun" in s1
        assert "-ntmpi 4" in s1
        assert "-pme gpu -npme 1" in s1

        s2 = _read_script(ws, "gromacs", "stmv_pme", "multi_rank_single_ntmpi")
        assert "mpirun" not in s2
        assert "-ntmpi 1" in s2
        assert "-pme gpu" in s2
        assert "-npme" not in s2

        w_pme = _read_script(ws, "gromacs", "water_gmx50", "water_pme_ntmpi4")
        assert "-pme gpu -npme 1" in w_pme
        assert "-ntmpi 4" in w_pme

        w_rf = _read_script(ws, "gromacs", "water_gmx50", "water_rf_ntmpi4")
        assert "-pme gpu" not in w_rf
        assert "-npme" not in w_rf


def test_gromacs_version_scoped_gpu_flags_and_env(workspace_name):
    """Verify -update gpu and GMX_ENABLE_DIRECT_GPU_COMM respect GROMACS versions."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="v2019",
            app_spec="gromacs@2019.6",
            extra_vars=["processes_per_node=1"],
            variants=["cuda=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="v2024",
            app_spec="gromacs@2024.6",
            extra_vars=["processes_per_node=1"],
            variants=["cuda=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="v2025",
            app_spec="gromacs@2025.3",
            extra_vars=["processes_per_node=1"],
            variants=["cuda=true"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        s2019 = _read_script(ws, "gromacs@2019.6", "water_gmx50", "v2019")
        assert "-update gpu" not in s2019
        assert "GMX_ENABLE_DIRECT_GPU_COMM" not in s2019

        s2024 = _read_script(ws, "gromacs@2024.6", "water_gmx50", "v2024")
        assert "-update gpu" in s2024
        assert "export GMX_ENABLE_DIRECT_GPU_COMM=1;" in s2024

        s2025 = _read_script(ws, "gromacs@2025.3", "water_gmx50", "v2025")
        assert "-update gpu" in s2025
        assert "GMX_ENABLE_DIRECT_GPU_COMM" not in s2025


def test_gromacs_cufftmp_and_nvshmem_env_vars(workspace_name):
    """Verify +cufftmp and +nvshmem env vars, including type=rf guard for cuFFTMp."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="cufftmp_pme",
            extra_vars=["processes_per_node=4", "type=pme", "npme=2"],
            variants=["cuda=true", "cufftmp=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="cufftmp_rf",
            extra_vars=["processes_per_node=4", "type=rf"],
            variants=["cuda=true", "cufftmp=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_bare",
            exp_name="cufftmp_bare_rf",
            extra_vars=["processes_per_node=4", "type=rf"],
            variants=["cuda=true", "cufftmp=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "hecbiosim",
            exp_name="cufftmp_hec",
            extra_vars=["processes_per_node=4"],
            variants=["cuda=true", "cufftmp=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="nvshmem_pme",
            extra_vars=["processes_per_node=4", "type=pme"],
            variants=["cuda=true", "nvshmem=true"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        cufftmp_pme = _read_script(ws, "gromacs", "water_gmx50", "cufftmp_pme")
        assert "export GMX_GPU_PME_DECOMPOSITION=1;" in cufftmp_pme
        assert "-pme gpu -npme 2" in cufftmp_pme

        cufftmp_rf = _read_script(ws, "gromacs", "water_gmx50", "cufftmp_rf")
        assert "GMX_GPU_PME_DECOMPOSITION" not in cufftmp_rf
        assert "-pme gpu" not in cufftmp_rf

        cufftmp_bare_rf = _read_script(
            ws, "gromacs", "water_bare", "cufftmp_bare_rf"
        )
        assert "GMX_GPU_PME_DECOMPOSITION" not in cufftmp_bare_rf
        assert "-pme gpu" not in cufftmp_bare_rf

        cufftmp_hec = _read_script(ws, "gromacs", "hecbiosim", "cufftmp_hec")
        assert "export GMX_GPU_PME_DECOMPOSITION=1;" in cufftmp_hec
        assert "-pme gpu -npme 1" in cufftmp_hec

        nvshmem_pme = _read_script(ws, "gromacs", "water_gmx50", "nvshmem_pme")
        assert "export GMX_ENABLE_NVSHMEM=1;" in nvshmem_pme
        assert "GMX_GPU_PME_DECOMPOSITION" not in nvshmem_pme


def test_gromacs_software_specs_concretize(workspace_name):
    """Verify spack concretize handles +mpi/+nvhpc/~mpi software specs and compilers."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="cpu_mpi",
            extra_vars=["processes_per_node=1", "env_name=env_cpu_mpi"],
            variants=["package_manager=spack"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="gpu_nvhpc",
            extra_vars=["processes_per_node=1", "env_name=env_gpu_nvhpc"],
            variants=["package_manager=spack", "cuda=true", "nvhpc=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="gpu_tmpi",
            extra_vars=["processes_per_node=1", "env_name=env_gpu_tmpi"],
            variants=["package_manager=spack", "cuda=true", "mpi=false"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="gpu_tmpi_nvhpc",
            extra_vars=["processes_per_node=1", "env_name=env_gpu_tmpi_nvhpc"],
            variants=[
                "package_manager=spack",
                "cuda=true",
                "mpi=false",
                "nvhpc=true",
            ],
        )
        workspace("concretize", global_args=global_args)
        with open(ws.config_file_path, encoding="utf-8") as f:
            cfg = f.read()
        assert "intel-oneapi-mpi@2021.17.2" in cfg
        assert "openmpi" not in cfg
        assert "{application::variant::mpi}" in cfg
        assert "{application::variant::cuda}" in cfg

        parsed = syaml.load(cfg)
        pkgs = parsed["ramble"]["software"]["packages"]
        envs = parsed["ramble"]["software"]["environments"]

        assert pkgs["gcc14"]["pkg_spec"] == "gcc@14.2.0"
        assert pkgs["nvhpc"]["pkg_spec"] == "nvhpc +mpi"
        assert pkgs["nvhpc"]["compiler_spec"] == "nvhpc"
        assert pkgs["nvhpc-mpi"]["pkg_spec"] == "nvhpc +mpi"
        assert (
            pkgs["gromacs-{application::gromacs::version}"]["compiler"]
            == "gcc14"
        )
        assert (
            pkgs["gromacs-{application::gromacs::version}-nvhpc"]["compiler"]
            == "nvhpc"
        )

        assert "intel-mpi" in envs["env_cpu_mpi"]["packages"]
        assert "nvhpc-mpi" not in envs["env_cpu_mpi"]["packages"]
        assert (
            "gromacs-{application::gromacs::version}"
            in envs["env_cpu_mpi"]["packages"]
        )

        assert "nvhpc-mpi" in envs["env_gpu_nvhpc"]["packages"]
        assert "intel-mpi" not in envs["env_gpu_nvhpc"]["packages"]
        assert (
            "gromacs-{application::gromacs::version}-nvhpc"
            in envs["env_gpu_nvhpc"]["packages"]
        )

        assert "intel-mpi" not in envs["env_gpu_tmpi"]["packages"]
        assert "nvhpc-mpi" not in envs["env_gpu_tmpi"]["packages"]
        assert (
            "gromacs-{application::gromacs::version}"
            in envs["env_gpu_tmpi"]["packages"]
        )

        assert "intel-mpi" not in envs["env_gpu_tmpi_nvhpc"]["packages"]
        assert "nvhpc-mpi" not in envs["env_gpu_tmpi_nvhpc"]["packages"]
        assert (
            "gromacs-{application::gromacs::version}-nvhpc"
            in envs["env_gpu_tmpi_nvhpc"]["packages"]
        )


def test_gromacs_gpu_aware_mpi_gating(workspace_name):
    """Verify GMX_FORCE_GPU_AWARE_MPI is set only for +mpi +nvhpc or +gpu_aware_mpi."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="cuda_mpi_no_nvhpc",
            extra_vars=["processes_per_node=2"],
            variants=["cuda=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="cuda_tmpi_no_nvhpc",
            extra_vars=["processes_per_node=1"],
            variants=["cuda=true", "mpi=false"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="cuda_tmpi_nvhpc",
            extra_vars=["processes_per_node=1"],
            variants=["cuda=true", "mpi=false", "nvhpc=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="cuda_mpi_nvhpc",
            extra_vars=["processes_per_node=2"],
            variants=["cuda=true", "nvhpc=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="cuda_mpi_optin",
            extra_vars=["processes_per_node=2"],
            variants=["cuda=true", "gpu_aware_mpi=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="cuda_mpi_nvhpc_optin",
            extra_vars=["processes_per_node=2"],
            variants=["cuda=true", "nvhpc=true", "gpu_aware_mpi=true"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        for exp in (
            "cuda_mpi_no_nvhpc",
            "cuda_tmpi_no_nvhpc",
            "cuda_tmpi_nvhpc",
        ):
            script = _read_script(ws, "gromacs", "water_gmx50", exp)
            assert "GMX_FORCE_GPU_AWARE_MPI" not in script

        for exp in (
            "cuda_mpi_nvhpc",
            "cuda_mpi_optin",
            "cuda_mpi_nvhpc_optin",
        ):
            script = _read_script(ws, "gromacs", "water_gmx50", exp)
            assert script.count("export GMX_FORCE_GPU_AWARE_MPI=1;") == 1


def test_gromacs_additional_args_decoupled(workspace_name):
    """Verify additional_args is decoupled from confout_args, pin_args, and ntomp_args."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name) as ws:
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="cpu_default",
            extra_vars=["processes_per_node=1", "n_threads=4"],
            variants=["cuda=false"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="gpu_custom_additional",
            extra_vars=[
                "processes_per_node=1",
                "n_threads=4",
                "additional_args=-maxh 0.5",
            ],
            variants=["cuda=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="gpu_no_pin",
            extra_vars=["processes_per_node=1", "n_threads=4", "pin_args="],
            variants=["cuda=true"],
        )
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            exp_name="gpu_keep_confout",
            extra_vars=[
                "processes_per_node=1",
                "n_threads=4",
                "confout_args=",
            ],
            variants=["cuda=true"],
        )
        workspace("setup", "--dry-run", global_args=global_args)

        cpu_default = _read_script(ws, "gromacs", "water_gmx50", "cpu_default")
        assert "-noconfout" in cpu_default
        assert "-pin on" not in cpu_default
        assert "-ntomp" not in cpu_default

        gpu_custom = _read_script(
            ws, "gromacs", "water_gmx50", "gpu_custom_additional"
        )
        assert "-ntomp 4" in gpu_custom
        assert "-pin on" in gpu_custom
        assert "-noconfout" in gpu_custom
        assert "-maxh 0.5" in gpu_custom

        gpu_no_pin = _read_script(ws, "gromacs", "water_gmx50", "gpu_no_pin")
        assert "-ntomp 4" in gpu_no_pin
        assert "-noconfout" in gpu_no_pin
        assert "-pin" not in gpu_no_pin

        gpu_keep = _read_script(
            ws, "gromacs", "water_gmx50", "gpu_keep_confout"
        )
        assert "-ntomp 4" in gpu_keep
        assert "-pin on" in gpu_keep
        assert "-noconfout" not in gpu_keep


@pytest.mark.parametrize(
    "app_spec,variants",
    [
        ("gromacs", ["nvhpc=true", "cuda=false"]),
        ("gromacs", ["gpu_aware_mpi=true", "cuda=false"]),
        ("gromacs", ["gpu_aware_mpi=true", "cuda=true", "mpi=false"]),
        ("gromacs", ["cufftmp=true", "cuda=false"]),
        ("gromacs", ["cufftmp=true", "cuda=true", "mpi=false"]),
        ("gromacs@2019.6", ["cufftmp=true", "cuda=true"]),
        ("gromacs", ["nvshmem=true", "cuda=false"]),
        ("gromacs", ["nvshmem=true", "cuda=true", "mpi=false"]),
        ("gromacs", ["nvshmem=true", "cufftmp=true", "cuda=true"]),
        ("gromacs@2019.6", ["nvshmem=true", "cuda=true"]),
    ],
)
def test_gromacs_variant_conflicts(workspace_name, app_spec, variants):
    """Verify incompatible gromacs variants/versions fail validation."""
    global_args = ["-w", workspace_name]
    with ramble.workspace.create(workspace_name):
        _add_gromacs_exp(
            workspace_name,
            "water_gmx50",
            app_spec=app_spec,
            extra_vars=["processes_per_node=1"],
            variants=variants,
        )
        with pytest.raises(ramble.error.ObjectValidationError):
            workspace("setup", "--dry-run", global_args=global_args)
