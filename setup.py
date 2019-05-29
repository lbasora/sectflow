from setuptools import setup

setup(
    name="sectflow",
    version="0.1",
    author="Luis Basora",
    url="https://github.com/lbasora/sectflow/",
    description="A trajectory clustering library to identify air traffic flow",
    license="MIT",
    packages=["sectflow"],
    install_requires=["numpy", "pandas", "sklearn", "traffic>=1.2.1b0"],
    dependency_links=[
        "https://github.com/xoolive/traffic/tarball/master#egg=traffic-1.2.1b0"
    ],
    python_requires=">=3.6",
)
