from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neural-spectral-gp",
    version="0.1.0",
    authors=[
        {"name": "Abdullah Karatas"},
        {"name": "Arsalan Jawaid"},
    ],
    author_email="",
    description="Learning Spectral Densities for Efficient Nonstationary Gaussian Process Simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AbdullahKaratas/neural-spectral-gp",
    project_urls={
        "Bug Tracker": "https://github.com/AbdullahKaratas/neural-spectral-gp/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
)
