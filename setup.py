#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="kernel-opt",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "typer>=0.9.0",
        "pyyaml>=6.0",
        "jinja2>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "kernel-opt=kernel_opt.cli:main",
            "kernel-opt-mcp=kernel_opt.mcp_server:run_server",
        ],
    },
    python_requires=">=3.10",
)



