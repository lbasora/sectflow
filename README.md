# sectflow
This is a python library to identify the air traffic flows in a ATC sector/airspace by applying a progressive 2D/3D trajectory clustering method based on DBSCAN.

The method is described in the paper [Occupancy Peak Estimation from Sector Geometry and Traffic Flow data](https://www.sesarju.eu/sites/default/files/documents/sid/2018/papers/SIDs_2018_paper_23.pdf). It has been developed as part of a larger framework to estimate the occupancy peak (capacity) of a ATC sector based on its traffic flow and geometric characteristics.

## Installation
Although **sectflow** is not directly dependent on the [traffic](https://github.com/xoolive/traffic) library, it has been designed to work closely integrated with it. See **traffic** installation instructions [here](https://github.com/xoolive/traffic).

The only necessary dependencies are [Numpy](https://www.numpy.org/) and [scikit-learn](https://scikit-learn.org/stable/).

To install **sectflow** and its dependencies run:

```
pip install git+https://github.com/lbasora/sectflow
```

## Use
The **demo_clustering** notebook illustrates how to use the library. Also, it is important to read the [clustering](https://traffic-viz.github.io/clustering.html) documentation in the **traffic** library to understand how both libraries work together.

The traffic features to be used for clustering are specified in the _features_ parameter of both the _TrajClust_ constructor and the _clustering_ method of the _Traffic_ class. 

Thus, traffic flows can be identified in 2D if only features _x_ an _y_ (or _latitude_ and _longitude_) are provided. Also, 3D flows can be identified by including _altitude_ (or _log_altitude_) in the parameter _features_. 

Please note that in any case  _x_ an _y_ are mandatory features, even if _latitude_ and _longitude_ are also provided. This is because the coordinate projections are needed for internal flow computation. If the traffic dataset does not contain the projected _x_ an _y_ coordinates, they will be automatically computed by the **traffic** library before calling **sectflow** _fit_ method.





