from setuptools import setup, find_packages

setup(
    name="parallel_runner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        # Add additional dependencies here if required.
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for processing a concatenated pandas DataFrame concurrently using an LLM service.",
    url="https://github.com/yourusername/parallel_runner",  # Replace with your actual GitHub URL.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
