"""Download alpha-beta-CROWN at a pinned version and make it importable.

Run once, inside the Python environment you want to use:
    python scripts/install_verifier.py

Needs: git and an internet connection.

We pin a specific commit rather than tracking master, so the professor
gets exactly the version this project was tested against. The commit
also fixes the auto_LiRPA submodule.

Everything (torch, torchvision, tqdm, onnx2pytorch, scipy, pandas, ...)
gets installed by pip from PyPI. Do NOT try to pre-install torch from
conda — abcrown 0.8.0 requires torch==2.11.0 exactly, which conda's
pytorch channel doesn't have on Win-64.
"""
import pathlib
import subprocess
import sys
import sysconfig

REPO = "https://github.com/Verified-Intelligence/alpha-beta-CROWN.git"
COMMIT = "b58d83ccb32469fc7ae9304fa1d33568ea76b00e"  # the version we tested
ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = ROOT / "third_party" / "alpha-beta-CROWN"


def run(command, **kwargs):
    """Print a command, then run it. Stop if it fails."""
    print("+", " ".join(str(part) for part in command))
    subprocess.check_call(command, **kwargs)


def main():
    # 1. Clone the verifier (with its auto_LiRPA submodule) at a fixed commit.
    if not TARGET.exists():
        run(["git", "clone", "--recursive", REPO, str(TARGET)])
    run(["git", "checkout", COMMIT], cwd=TARGET)
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=TARGET)

    # 2. Install the verifier and all its Python deps from PyPI.
    #    This is what pulls torch==2.11.0 etc.
    run([sys.executable, "-m", "pip", "install", "-e", str(TARGET)])

    # 3. Remove a stray auto_LiRPA install if there is one — it would
    #    shadow the submodule we just cloned.
    subprocess.call([sys.executable, "-m", "pip",
                     "uninstall", "-y", "auto-lirpa"])

    # 4. Write a .pth file so Python can find the verifier's folders
    #    without us having to fiddle with sys.path everywhere.
    pth_file = pathlib.Path(sysconfig.get_paths()["purelib"]) / "verifierprobe_paths.pth"
    pth_file.write_text(f"{TARGET / 'auto_LiRPA'}\n{TARGET}\n")
    print("wrote", pth_file)

    # 5. Sanity check: can we import the verifier?
    run([sys.executable, "-c",
         "from abcrown import ABCrownSolver; print('alpha-beta-CROWN import OK')"])


if __name__ == "__main__":
    main()