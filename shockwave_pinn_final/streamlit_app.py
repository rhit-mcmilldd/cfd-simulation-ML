"""
streamlit_app.py
----------------
Browser-based UI for the Shockwave PINN pipeline.

This app exposes the existing synthetic and SU2 pipeline commands in a
forward-facing interface, so you can adjust flow conditions and inspect
the generated output images without using the CLI directly.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_IMAGES = [
    "mach_contour.png",
    "flow_field.png",
    "error_field.png",
    "training_loss.png",
]


def format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_subprocess(command: list[str], cwd: Path) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    output_lines = []
    if process.stdout is not None:
        for line in process.stdout:
            output_lines.append(line)
            st.session_state.log_area.text("".join(output_lines))

    process.wait()
    return process.returncode, "".join(output_lines)


def show_output_images(output_dir: Path):
    cols = st.columns(2)
    for idx, image_name in enumerate(OUTPUT_IMAGES):
        path = output_dir / image_name
        if path.exists():
            with cols[idx % 2]:
                st.image(str(path), caption=image_name, use_column_width=True)


def build_main_command(
    mode: str,
    task: str,
    data_dir: str,
    output_dir: str,
    mach_inf: float,
    wedge_angle_deg: float,
    p_inf: float,
    T_inf: float,
    rho_inf: float,
    n_adam: int,
    n_lbfgs: int,
    lr: float,
    warmup_steps: int,
    checkpoint: str | None = None,
    extra_args: str = "",
) -> list[str]:
    cmd = [sys.executable, str(PROJECT_ROOT / "main.py"), "--task", task, "--mode", mode]
    cmd.extend(["--mach_inf", str(mach_inf)])
    cmd.extend(["--wedge_angle_deg", str(wedge_angle_deg)])
    cmd.extend(["--p_inf", str(p_inf)])
    cmd.extend(["--T_inf", str(T_inf)])
    cmd.extend(["--rho_inf", str(rho_inf)])
    cmd.extend(["--output_dir", output_dir])
    cmd.extend(["--data_dir", data_dir])
    if task in ("train", "train_visualize"):
        cmd.extend(["--n_adam", str(n_adam)])
        cmd.extend(["--n_lbfgs", str(n_lbfgs)])
        cmd.extend(["--lr", str(lr)])
        cmd.extend(["--warmup_steps", str(warmup_steps)])
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    return cmd


def main():
    st.set_page_config(page_title="Shockwave PINN UI", layout="wide")

    st.title("Shockwave PINN UI")
    st.write(
        "Use this interface to run the synthetic or SU2 PINN workflow, "
        "adjust flow conditions, and preview generated output images."
    )

    with st.sidebar:
        st.header("Pipeline settings")
        mode = st.selectbox("Mode", ["synthetic", "su2"], index=0)
        task = st.selectbox("Task", ["train", "train_visualize", "visualize"], index=1)
        output_dir = st.text_input("Output directory", str(DEFAULT_OUTPUT_DIR))
        st.write("---")
        st.header("Flow conditions")
        mach_inf = st.number_input("Mach number (M∞)", value=2.5, min_value=1.01, step=0.1)
        wedge_angle_deg = st.number_input("Wedge / AoA (degrees)", value=15.0, min_value=0.1, max_value=45.0, step=0.5)
        p_inf = st.number_input("Free-stream pressure (Pa)", value=101325.0, min_value=1.0, step=100.0, format="%.1f")
        T_inf = st.number_input("Free-stream temperature (K)", value=300.0, min_value=1.0, step=1.0, format="%.1f")
        rho_inf = st.number_input("Free-stream density (kg/m³)", value=1.225, min_value=0.01, step=0.01, format="%.4f")
        st.write("---")

        if task in ("train", "train_visualize"):
            st.header("Training options")
            n_adam = st.number_input("Adam steps", value=5000, min_value=0, step=100)
            n_lbfgs = st.number_input("L-BFGS iterations", value=300, min_value=0, step=10)
            lr = st.number_input("Learning rate", value=1e-3, min_value=1e-8, format="%.6f")
            warmup_steps = st.number_input("Physics warmup steps", value=500, min_value=0, step=50)
        else:
            n_adam = 5000
            n_lbfgs = 300
            lr = 1e-3
            warmup_steps = 500

        extra_args = st.text_input("Extra main.py arguments", value="")
        data_dir = st.text_input("SU2 data file or directory", "su2_output.csv")
        checkpoint = st.text_input("Checkpoint path (resume training or visualize)", value="")

    if task == "visualize" and not checkpoint:
        st.warning("Visualization mode requires a saved checkpoint path.")

    if st.button("Run pipeline"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        st.session_state.log_area = st.empty()
        st.session_state.log_area.text("Preparing command...\n")

        command = build_main_command(
            mode=mode,
            task=task,
            data_dir=data_dir,
            output_dir=str(output_path),
            mach_inf=mach_inf,
            wedge_angle_deg=wedge_angle_deg,
            p_inf=p_inf,
            T_inf=T_inf,
            rho_inf=rho_inf,
            n_adam=n_adam,
            n_lbfgs=n_lbfgs,
            lr=lr,
            warmup_steps=warmup_steps,
            checkpoint=checkpoint if checkpoint else None,
            extra_args=extra_args,
        )

        st.session_state.log_area.text(f"Command:\n{format_command(command)}\n\n")
        returncode, _ = run_subprocess(command, cwd=PROJECT_ROOT)

        if returncode == 0:
            st.success("Pipeline completed successfully.")
        else:
            st.error(f"Pipeline failed with return code {returncode}.")

        st.write("---")
        st.header("Generated output")
        show_output_images(output_path)

    st.sidebar.write("---")
    st.sidebar.markdown(
        "**Tips:** Use `--wedge-angle-deg` to adjust the wedge/AoA geometry and `--mach-inf` to change the free-stream Mach number. For SU2 mode, provide the CSV output folder or file path in the data directory field."
    )


if __name__ == "__main__":
    main()
