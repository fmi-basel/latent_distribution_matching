
import os

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

KW = ["artificial intelligence", "deep learning", "unsupervised learning", "contrastive learning"]

REQUIREMENTS_FILE = os.path.join(os.path.dirname(__file__), "requirements.txt")
with open(REQUIREMENTS_FILE) as fo:
    REQUIREMENTS = [str(req) for req in parse_requirements(fo.readlines())]

EXTRA_REQUIREMENTS = {
    "h5": ["h5py"],
}


def parse_requirements(path):
    with open(path) as f:
        requirements = [p.strip().split()[-1] for p in f.readlines()]
    return requirements


setup(
    name="probssl",
    packages=find_packages(exclude=["bash_files", "docs", "downstream", "tests", "zoo"]),
    version="0.0.0",
    license="MIT",
    author="",
    author_email="",
    url="",
    keywords=KW,
    install_requires=REQUIREMENTS,
    extras_require=EXTRA_REQUIREMENTS,
    dependency_links=["https://developer.download.nvidia.com/compute/redist"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
)
