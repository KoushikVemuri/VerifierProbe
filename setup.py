from setuptools import setup, find_packages

setup(
    name="verifier_probe",
    version="0.1.0",
    description="Adversarial probing for alpha,beta-CROWN",
    packages=find_packages(include=["verifier_probe*"]),
    install_requires=[
        "torch>=2.1,<3.0",
        "numpy>=1.24,<3.0",
        "sympy>=1.12",
    ],
    python_requires=">=3.11",
    # alpha-beta-CROWN is not a pip dependency: it is cloned and installed
    # from source by scripts/install_verifier.py. See README.md.
)