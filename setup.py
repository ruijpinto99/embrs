from setuptools import setup, find_packages

setup(
    name="embrs",
    version="0.0.13",
    author="Rui de Gouvea Pinto",
    author_email="rjdp3@gatech.edu",
    description="A fire simulation tool optimized for autonomous firefighting systems development",
    url="https://github.com/AREAL-GT/embrs",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.0",
        "shapely>=2.0",
		"matplotlib>=3.0",
		"PyQt5>=5.0",
		"tqdm>=4.0",
		"pandas>=2.0",
		"rasterio>=1.0",
		"requests>=2.0",
		"pyproj>=3.0",
		"utm>=0.6",
		"scipy>=1.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
	entry_points={
        "console_scripts": [
            "run_embrs_sim = embrs.main:main",
            "create_embrs_map = embrs.map_generator:main",
            "create_embrs_wind = embrs.wind_forecast_generator:main",
            "run_embrs_viz = embrs.visualization_tool:main"
        ],
	},
)
