from setuptools import setup, find_packages

setup(
    name="mfreeze",
    version="0.0.1",
    url="https://github.com/Ruairi-osul/mfreeze.git",
    author="Ruairi O'Sullivan",
    author_email="ruairi.osullivan.work@gmail.com",
    description="Detect freezes",
    packages=find_packages(),
    install_requires=[
        "pandas==0.23.0",
        "matplotlib==3.1.1",
        # "opencv==3.4.3",
        "jupyter==1.0.0",
        "holoviews==1.12.3",
        "scipy==1.2.1",
        "bokeh==1.2.0",
        "opencv-python",
    ],
)
