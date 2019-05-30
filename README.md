# sectflow
A python library to identify the air traffic flows in a ATC sector/airspace by applying a progressive 2D/3D trajectory clustering method based on DBSCAN.

The method is described in the paper [Occupancy Peak Estimation from Sector Geometry and Traffic Flow data](https://www.sesarju.eu/sites/default/files/documents/sid/2018/papers/SIDs_2018_paper_23.pdf). It has been used to characterise the traffic crossing a sector in order to estimate the occupancy peak of an ATC sector.

## Installation
Although **sectflow** is not directly dependend on the **[traffic] (https://github.com/xoolive/traffic) library**, it has been designed to work closely integrated with it. Therefore, it is recommended to install **traffic** first (see installation instructions [here](https://github.com/xoolive/traffic)).

Otherwise, the only necessary dependencies are [Numpy](https://www.numpy.org/) and [scikit-learn](https://scikit-learn.org/stable/), which will be installed when running:

```
pip install git+https://github.com/lbasora/sectflow
```
## Use
The **demo_clusetring** notebook illustrates how to use the library.  




