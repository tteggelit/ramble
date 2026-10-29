# Copyright 2022-2026 The Ramble Authors
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

import glob
import os
import re

from ramble.appkit import *
from ramble.expander import Expander


class Relion(ExecutableApplication):
    """RELION (for REgularised LIkelihood OptimisatioN) is a stand-alone
    computer program that employs an empirical Bayesian approach to refinement
    of (multiple) 3D reconstructions or 2D class averages in electron
    cryo-microscopy (cryo-EM).

    https://relion.readthedocs.io/en/release-5.0/index.html
    """

    name = "relion"

    maintainers("tteggelit")

    tags("cryo-em", "bioinformatics", "hpc-benchmark")

    version("5.0.1", description="Version 5.0.1 of RELION", preferred=True)
    version("4.0.1", description="Version 4.0.1 of RELION")
    version("4.0.0", description="Version 4.0.0 of RELION")
    version("3.1.4", description="Version 3.1.4 of RELION")
    version("3.1.3", description="Version 3.1.3 of RELION")
    version("3.1.2", description="Version 3.1.2 of RELION")
    version("3.1.1", description="Version 3.1.1 of RELION")
    version("3.1.0", description="Version 3.1.0 of RELION")
    version("3.0.8", description="Version 3.0.8 of RELION")
    version("3.0.7", description="Version 3.0.7 of RELION")

    variant(
        "cuda",
        default=False,
        values=[True, False],
        description="CUDA GPU acceleration support variant",
    )

    variant(
        "nvhpc",
        default=False,
        values=[True, False],
        description="NVIDIA HPC SDK support variant",
    )

    with when("+nvhpc"):
        conflict(
            "~cuda",
            msg="The nvhpc variant requires cuda to be enabled (+cuda)",
        )

    variant(
        "altcpu",
        default=False,
        values=[True, False],
        description="Use vectorized CPU acceleration (SIMD/TBB)",
    )

    with when("+altcpu"):
        conflict(
            "+cuda",
            msg="The altcpu variant cannot be enabled with cuda (+cuda)",
        )

    variant(
        "mklfft",
        default=False,
        values=[True, False],
        description="Use Intel MKL rather than FFTW for FFT",
    )

    variant(
        "double",
        default=True,
        values=[True, False],
        description="Double precision CPU code",
    )

    variant(
        "double-gpu",
        default=False,
        values=[True, False],
        description="Double precision GPU code",
    )

    variant(
        "python",
        default=False,
        values=[True, False],
        description="Include py-relion Python tools for RELION 5+",
    )

    with when("package_manager_family=spack"):
        define_compiler("gcc14", pkg_spec="gcc@14.2.0")

        software_spec(
            "intel-mpi",
            pkg_spec="intel-oneapi-mpi@2021.17.2",
            when=["~nvhpc"],
        )

        software_spec(
            "openmpi",
            pkg_spec="openmpi",
            when=["+nvhpc"],
        )

        with default_args(compiler="gcc14"):
            with when("application_version@5.0.0:"):
                with when("~altcpu"):
                    software_spec(
                        "relion-{application::relion::version}",
                        pkg_spec="relion@{application::relion::version} ~gui "
                        + "{application::variant::cuda} "
                        + "{application::variant::double} "
                        + "{application::variant::double-gpu} "
                        + "{application::variant::mklfft} "
                        + "{application::variant::python}",
                    )

                with when("+altcpu"):
                    software_spec(
                        "relion-altcpu-{application::relion::version}",
                        pkg_spec="relion@{application::relion::version} ~gui "
                        + "{application::variant::cuda} "
                        + "{application::variant::altcpu} "
                        + "{application::variant::double} "
                        + "{application::variant::double-gpu} "
                        + "{application::variant::mklfft} "
                        + "{application::variant::python}",
                    )

            with when("application_version@:4.0.1"):
                with when("~altcpu"):
                    software_spec(
                        "relion4-{application::relion::version}",
                        pkg_spec="relion@{application::relion::version} ~gui "
                        + "{application::variant::cuda} "
                        + "{application::variant::double} "
                        + "{application::variant::double-gpu} "
                        + "{application::variant::mklfft}",
                    )

                with when("+altcpu"):
                    software_spec(
                        "relion4-altcpu-{application::relion::version}",
                        pkg_spec="relion@{application::relion::version} ~gui "
                        + "{application::variant::cuda} "
                        + "{application::variant::altcpu} "
                        + "{application::variant::double} "
                        + "{application::variant::double-gpu} "
                        + "{application::variant::mklfft}",
                    )

        required_package("relion")

    input_file(
        "plasmodium_ribosome",
        url="ftp://ftp.mrc-lmb.cam.ac.uk/pub/scheres/relion_benchmark.tar.gz",
        description="Plasmodium falciparum 80S ribosome standard RELION benchmark dataset (EMD-2660, 105,247 particles)",
    )

    executable(
        "print-version",
        template="{relion_bin} --version",
        use_mpi=False,
        output_capture=OUTPUT_CAPTURE.ALL,
    )

    executable(
        "stage-links",
        template=[
            "ln -sfn {input_path}/Particles {experiment_run_dir}/Particles",
            "ln -sfn {input_path}/emd_2660.map {experiment_run_dir}/emd_2660.map",
        ],
        use_mpi=False,
    )

    executable(
        "execute-class2d",
        template="{relion_bin} "
        + "--i {star_file} "
        + "--o {output_prefix} "
        + "--ctf "
        + "--iter {n_iter} "
        + "--tau2_fudge 2 "
        + "--particle_diameter 360 "
        + "--K {num_classes} "
        + "--flatten_solvent "
        + "--zero_mask "
        + "--oversampling 1 "
        + "--psi_step 6 "
        + "--offset_range 5 "
        + "--offset_step 2 "
        + "--norm "
        + "--scale "
        + "--random_seed 0 "
        + "--j {n_threads} "
        + "--pool {pool} "
        + "--pad 2 "
        + "{dont_combine_weights_via_disc} "
        + "{preread_flags} "
        + "{scratch_flags} "
        + "{accel_flags} "
        + "{additional_args}",
        use_mpi=True,
        output_capture=OUTPUT_CAPTURE.ALL,
    )

    executable(
        "execute-class3d",
        template="{relion_bin} "
        + "--i {star_file} "
        + "--o {output_prefix} "
        + "--ref {ref_map} "
        + "--firstiter_cc "
        + "--ini_high 60 "
        + "--ctf "
        + "--iter {n_iter} "
        + "--tau2_fudge 4 "
        + "--particle_diameter 360 "
        + "--K {num_classes} "
        + "--flatten_solvent "
        + "--zero_mask "
        + "--oversampling 1 "
        + "--healpix_order 2 "
        + "--offset_range 5 "
        + "--offset_step 2 "
        + "--sym C1 "
        + "--norm "
        + "--scale "
        + "--random_seed 0 "
        + "--j {n_threads} "
        + "--pool {pool} "
        + "--pad 2 "
        + "{dont_combine_weights_via_disc} "
        + "{preread_flags} "
        + "{scratch_flags} "
        + "{accel_flags} "
        + "{additional_args}",
        use_mpi=True,
        output_capture=OUTPUT_CAPTURE.ALL,
    )

    executable(
        "execute-refine3d",
        template="{relion_bin} "
        + "--auto_refine "
        + "--split_random_halves "
        + "--i {star_file} "
        + "--o {output_prefix} "
        + "--ref {ref_map} "
        + "--firstiter_cc "
        + "--ini_high 60 "
        + "--ctf "
        + "--particle_diameter 360 "
        + "--flatten_solvent "
        + "--zero_mask "
        + "--oversampling 1 "
        + "--healpix_order 2 "
        + "--auto_local_healpix_order 4 "
        + "--offset_range 5 "
        + "--offset_step 2 "
        + "--sym C1 "
        + "--low_resol_join_halves 40 "
        + "--norm "
        + "--scale "
        + "--random_seed 0 "
        + "--j {n_threads} "
        + "--pool {pool} "
        + "--pad 2 "
        + "{dont_combine_weights_via_disc} "
        + "{preread_flags} "
        + "{scratch_flags} "
        + "{accel_flags} "
        + "{additional_args}",
        use_mpi=True,
        output_capture=OUTPUT_CAPTURE.ALL,
    )

    workload(
        "class2d",
        executables=["print-version", "stage-links", "execute-class2d"],
        input="plasmodium_ribosome",
    )

    workload(
        "class3d",
        executables=["print-version", "stage-links", "execute-class3d"],
        input="plasmodium_ribosome",
    )

    workload(
        "refine3d",
        executables=["print-version", "stage-links", "execute-refine3d"],
        input="plasmodium_ribosome",
    )

    workload_group(
        "all_workloads",
        workloads=[
            "class2d",
            "class3d",
            "refine3d",
        ],
    )

    workload_group(
        "classification_workloads",
        workloads=[
            "class2d",
            "class3d",
        ],
    )

    workload_group(
        "3d_workloads",
        workloads=[
            "class3d",
            "refine3d",
        ],
    )

    workload_variable(
        "relion_bin",
        default="relion_refine_mpi",
        description="RELION refinement executable name",
        workload_group="all_workloads",
    )

    workload_variable(
        "input_path",
        default="{plasmodium_ribosome}",
        description="Path to the unpacked RELION benchmark dataset",
        workload_group="all_workloads",
    )

    workload_variable(
        "star_file",
        default="Particles/shiny_2sets.star",
        description="Input particle STAR file relative to experiment_run_dir",
        workload_group="all_workloads",
    )

    workload_variable(
        "output_prefix",
        default="run",
        description="Output filename prefix for RELION run files",
        workload_group="all_workloads",
    )

    workload_variable(
        "pool",
        default="100",
        description="Number of particles pooled together per batch for CPU/GPU processing",
        workload_group="all_workloads",
    )

    workload_variable(
        "dont_combine_weights_via_disc",
        default="--dont_combine_weights_via_disc",
        values=["--dont_combine_weights_via_disc", ""],
        description="Send large arrays of summed weights via MPI instead of writing to disk",
        workload_group="all_workloads",
    )

    workload_variable(
        "preread_flags",
        default="",
        values=["", "--preread_images"],
        description="Pre-read all particle images into RAM on the MPI leader rank",
        workload_group="all_workloads",
    )

    workload_variable(
        "scratch_flags",
        default="",
        description="Optional --scratch_dir <path> flag to copy particles to fast local scratch",
        workload_group="all_workloads",
    )

    workload_variable(
        "accel_flags",
        default="",
        description="Hardware acceleration flags (--gpu for CUDA, --cpu for altcpu SIMD)",
        workload_group="all_workloads",
    )

    with when("+cuda"):
        workload_variable(
            "gpu_ids",
            default="",
            description="GPU device ID mapping (e.g., '0:1:2:3') or empty string for automatic assignment",
            workload_group="all_workloads",
        )

        workload_variable(
            "accel_flags",
            default='--gpu "{gpu_ids}"',
            description="Enable GPU offloading for RELION refinement",
            workload_group="all_workloads",
        )

    with when("+altcpu"):
        workload_variable(
            "accel_flags",
            default="--cpu",
            description="Enable vectorized CPU acceleration (--cpu) for RELION refinement",
            workload_group="all_workloads",
        )

    workload_variable(
        "additional_args",
        default="",
        description="Additional command-line arguments passed to relion_refine_mpi",
        workload_group="all_workloads",
    )

    workload_variable(
        "n_iter",
        default="25",
        description="Number of expectation-maximization iterations",
        workload_group="classification_workloads",
    )

    workload_variable(
        "num_classes",
        default="200",
        description="Number of 2D classes (K)",
        workload="class2d",
    )

    workload_variable(
        "ref_map",
        default="emd_2660.map:mrc",
        description="Initial 3D reference map",
        workload_group="3d_workloads",
    )

    workload_variable(
        "num_classes",
        default="6",
        description="Number of 3D classes (K)",
        workload="class3d",
    )

    figure_of_merit(
        "RELION version",
        fom_regex=r"RELION version:\s*(?P<version>\S+)",
        group_name="version",
        units="",
        fom_type=FomType.INFO,
    )

    figure_of_merit(
        "Precision",
        fom_regex=r"Precision:\s*(?P<precision>.*)",
        group_name="precision",
        units="",
        fom_type=FomType.INFO,
    )

    stats_file_str = os.path.join(
        Expander.expansion_str("experiment_run_dir"), "relion_stats.out"
    )

    iter_fom_regex = (
        r"Iteration\s+(?P<iter>[0-9]+):\s+"
        + r"e_step_time\s*=\s*(?P<e_step_time>[0-9]+\.[0-9]+)\s*s"
        + r"(?:,\s*wall_time\s*=\s*(?P<iter_wall_time>[0-9]+\.[0-9]+)\s*s)?"
    )

    figure_of_merit_context(
        "Expectation iteration",
        regex=iter_fom_regex,
        output_format="Iteration {iter}",
    )

    figure_of_merit(
        "Iteration E-step time",
        log_file=stats_file_str,
        fom_regex=iter_fom_regex,
        group_name="e_step_time",
        units="s",
        fom_type=FomType.TIME,
        contexts=["Expectation iteration"],
    )

    figure_of_merit(
        "Iteration wall time",
        log_file=stats_file_str,
        fom_regex=iter_fom_regex,
        group_name="iter_wall_time",
        units="s",
        fom_type=FomType.TIME,
        contexts=["Expectation iteration"],
    )

    figure_of_merit(
        "Completed iterations",
        log_file=stats_file_str,
        fom_regex=r"Completed iterations:\s+(?P<completed_iters>[0-9]+)",
        group_name="completed_iters",
        units="",
        fom_type=FomType.MEASURE,
    )

    figure_of_merit(
        "Total E-step time",
        log_file=stats_file_str,
        fom_regex=r"Total E-step time:\s+(?P<total_e_time>[0-9]+\.[0-9]+)\s+s",
        group_name="total_e_time",
        units="s",
        fom_type=FomType.TIME,
    )

    figure_of_merit(
        "Average E-step time",
        log_file=stats_file_str,
        fom_regex=r"Average E-step time:\s+(?P<avg_e_time>[0-9]+\.[0-9]+)\s+s",
        group_name="avg_e_time",
        units="s",
        fom_type=FomType.TIME,
    )

    figure_of_merit(
        "Min E-step time",
        log_file=stats_file_str,
        fom_regex=r"Min E-step time:\s+(?P<min_e_time>[0-9]+\.[0-9]+)\s+s",
        group_name="min_e_time",
        units="s",
        fom_type=FomType.TIME,
    )

    figure_of_merit(
        "Max E-step time",
        log_file=stats_file_str,
        fom_regex=r"Max E-step time:\s+(?P<max_e_time>[0-9]+\.[0-9]+)\s+s",
        group_name="max_e_time",
        units="s",
        fom_type=FomType.TIME,
    )

    figure_of_merit(
        "Total iteration wall time",
        log_file=stats_file_str,
        fom_regex=r"Total iteration wall time:\s+(?P<total_wall_time>[0-9]+\.[0-9]+)\s+s",
        group_name="total_wall_time",
        units="s",
        fom_type=FomType.TIME,
    )

    figure_of_merit(
        "Average iteration wall time",
        log_file=stats_file_str,
        fom_regex=r"Average iteration wall time:\s+(?P<avg_wall_time>[0-9]+\.[0-9]+)\s+s",
        group_name="avg_wall_time",
        units="s",
        fom_type=FomType.TIME,
    )

    with when("workload=refine3d"):
        figure_of_merit(
            "Final resolution",
            fom_regex=r"Auto-refine: \+ Final resolution \(.*\) is:\s+(?P<final_res>[0-9]+(?:\.[0-9]+)?)",
            group_name="final_res",
            units="Angstroms",
            fom_type=FomType.MEASURE,
        )

    with when("workload_group=classification_workloads"):
        success_criteria(
            "Completed Iterations",
            mode="fom_comparison",
            fom_name="Completed iterations",
            formula="int({value}) == int({n_iter})",
        )

    with when("workload=refine3d"):
        success_criteria(
            "Completed Iterations",
            mode="fom_comparison",
            fom_name="Completed iterations",
            formula="int({value}) > 0",
        )

    success_criteria(
        "No RELION Fatal Error",
        mode="string",
        anti_match=r"(?:RELION_ERROR|^ERROR:)",
    )

    def _prepare_analysis(self, workspace, app_inst=None):
        """Parse RELION iteration timings from log output and checkpoint files."""
        exp_dir = self.expander.expand_var_name("experiment_run_dir")
        log_path = self.expander.expand_var_name("log_file")
        output_prefix = self.expander.expand_var("{output_prefix}")

        iter_header_re = re.compile(
            r"^\s*Expectation iteration\s+(?P<iter>[0-9]+)"
        )
        progress_done_re = re.compile(
            r"^\s*(?P<elapsed>[0-9]+(?:\.[0-9]+)?)/(?P<total>[0-9]+(?:\.[0-9]+)?)\s+"
            + r"(?P<unit>sec|min|hrs)\s+\.+~~\(,_,\">"
        )
        unit_scale = {"sec": 1.0, "min": 60.0, "hrs": 3600.0}

        e_step_times = {}
        if os.path.isfile(log_path):
            current_iter = None
            with open(log_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m_iter = iter_header_re.match(line)
                    if m_iter:
                        current_iter = int(m_iter.group("iter"))
                        continue

                    if current_iter is not None:
                        m_prog = progress_done_re.match(line)
                        if m_prog:
                            val = float(m_prog.group("total"))
                            unit = m_prog.group("unit")
                            e_step_times[current_iter] = val * unit_scale[unit]
                            current_iter = None

        # Compute full iteration wall times from checkpoint *_itNNN_optimiser.star mtimes if available
        iter_mtimes = {}
        star_pattern = os.path.join(
            exp_dir, f"{output_prefix}_it*_optimiser.star"
        )
        star_iter_re = re.compile(
            rf"{re.escape(output_prefix)}_it(?P<iter>[0-9]+)_optimiser\.star$"
        )
        for path in glob.glob(star_pattern):
            m_star = star_iter_re.search(os.path.basename(path))
            if m_star and os.path.isfile(path):
                iter_mtimes[int(m_star.group("iter"))] = os.path.getmtime(path)

        iter_wall_times = {}
        for it in sorted(iter_mtimes.keys()):
            if it > 0 and (it - 1) in iter_mtimes:
                dt = iter_mtimes[it] - iter_mtimes[it - 1]
                if dt >= 0.0:
                    iter_wall_times[it] = dt

        all_iters = sorted(
            set(e_step_times.keys()) | set(iter_wall_times.keys())
        )
        if all_iters and os.path.isdir(exp_dir):
            stats_path = os.path.join(exp_dir, "relion_stats.out")
            with open(stats_path, "w+", encoding="utf-8") as f:
                for it in all_iters:
                    e_val = e_step_times.get(it, iter_wall_times.get(it, 0.0))
                    if it in iter_wall_times:
                        f.write(
                            f"Iteration {it}: e_step_time = {e_val:.4f} s, "
                            f"wall_time = {iter_wall_times[it]:.4f} s\n"
                        )
                    else:
                        f.write(
                            f"Iteration {it}: e_step_time = {e_val:.4f} s\n"
                        )

                f.write(f"Completed iterations: {len(all_iters)}\n")

                if e_step_times:
                    vals = list(e_step_times.values())
                    total_e = sum(vals)
                    avg_e = total_e / len(vals)
                    f.write(f"Total E-step time: {total_e:.4f} s\n")
                    f.write(f"Average E-step time: {avg_e:.4f} s\n")
                    f.write(f"Min E-step time: {min(vals):.4f} s\n")
                    f.write(f"Max E-step time: {max(vals):.4f} s\n")

                if iter_wall_times:
                    w_vals = list(iter_wall_times.values())
                    total_w = sum(w_vals)
                    avg_w = total_w / len(w_vals)
                    f.write(f"Total iteration wall time: {total_w:.4f} s\n")
                    f.write(f"Average iteration wall time: {avg_w:.4f} s\n")
