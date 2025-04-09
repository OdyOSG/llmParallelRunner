from setuptools import setup, find_packages

setup(
    name="parallel_runner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        # Add additional dependencies here as required (e.g., if 'llmInvocation' is available on PyPI).
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for processing multiple pandas DataFrames concurrently using an LLM service.",
    url="https://github.com/yourusername/parallel_runner",  # Replace with your actual GitHub repository URL.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
