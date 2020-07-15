from setuptools import setup, find_packages

with open("requirements.txt") as f:
    reqs = f.read().splitlines()

setup(
    name="mfreeze",
    version="0.0.1",
    url="https://github.com/Ruairi-osul/mfreeze.git",
    author="Ruairi O'Sullivan",
    author_email="ruairi.osullivan.work@gmail.com",
    description="Detect freezes",
    project_urls={"Source": "https://github.com/Ruairi-osul/ezbootstrap"},
    packages=find_packages(),
    python_requires=">=3.3",
    install_requires=reqs,
)
