from setuptools import setup, find_packages

setup(
    name="stockraj",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "yfinance",
        "scikit-learn",
        "tensorflow",
        "plotly",
        "streamlit",
        "ta"
    ]
) 