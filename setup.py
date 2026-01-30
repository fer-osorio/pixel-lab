from setuptools import setup, find_packages

setup(
    name="image-toolkit",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "pillow>=9.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0"
    ],
    python_requires=">=3.8"
)
