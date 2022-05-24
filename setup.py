import os
from pathlib import Path
from setuptools import find_packages, setup
import subprocess

import torch

PACKAGE_NAME = "sota-zoo"

TORCH_VERSION = [int(x) for x in torch.__version__.split(".")[:2]]
assert TORCH_VERSION >= [1, 11], "Requires PyTorch >= 1.11"


def get_version() -> str:
    cwd = Path(__file__).parent

    with open(cwd / "version.txt") as f:
        version = f.readline().strip()

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
    except Exception:
        sha = "Unknown"

    if os.getenv("STZOO_BUILD_VERSION"):
        version = os.getenv("STZOO_BUILD_VERSION")
    elif sha != "Unknown":
        version += "+" + sha[:7]

    # write version.py
    with open(cwd / "sota_zoo" / "version.py", "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")

    return version


if __name__ == "__main__":
    version = get_version()

    setup(
        name=PACKAGE_NAME,
        version=version,
        author="Ming Yang",
        author_email="ymviv@qq.com",
        url="https://github.com/vivym/sota-zoo",
        download_url="https://github.com/vivym/sota-zoo/tags",
        description="Sota Zoo",
        long_description=Path("README.md").read_text(),
        packages=find_packages(exclude=("tests",)),
        package_data={"sota_zoo": ["*.dll", "*.so", "*.dylib"]},
        zip_safe=False,
        python_requires=">=3.9",
        install_requires=[
        ],
    )
