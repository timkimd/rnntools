from setuptools import setup, find_namespace_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "jax",
    "equinox",
    "optax",
]

setup(
    name="findir",
    version="0.0.1",
    description="JAX implementation of FINDIR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    extras_require={"dev": ["coverage-badge"]},
    python_requires=">=3.6,",
    packages=find_namespace_packages(),
    include_package_data=True,
    url="https://github.com/timkimd/findir",
    author="Tim Kim",
    zip_safe=False,
    entry_points={},
)
