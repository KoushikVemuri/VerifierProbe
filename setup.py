from setuptools import setup, find_packages

setup(
    name="verifier_probe",
    version="0.1.0",
    description="Adversarial probing and intermediate-state dumping for the alpha,beta-CROWN verifier",
    packages=find_packages(include=["verifier_probe*"]),
    install_requires=[
        "torch==2.11.0",
        "torchvision==0.26.0",
        "numpy==2.4.6",
        "sympy==1.14.0",
    ],
    extras_require={"dev": ["pytest==9.2.0"]},
    python_requires=">=3.11",
    # alpha-beta-CROWN is not a pip dependency: it is cloned and installed
    # from source by scripts/install_verifier.py. See README.md.
)