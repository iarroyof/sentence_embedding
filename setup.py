from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="sts_package",
    version="0.1.0",
    author="Your Name", # Replace with actual author
    author_email="your.email@example.com", # Replace with actual email
    description="A package for calculating sentence similarity using word embeddings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sts_package", # Replace with actual URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Assuming MIT, change if different
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "sts=sts_package.sts:main",
        ],
    },
)
