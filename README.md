# README #

This repo is a three part series of intermediate data visualization concepts in Python.

* [Session 1](#markdown-header-session-1)

* [Session 2](#markdown-header-session-2)

* [Session 3](#markdown-header-session-3)

Clone the repo to access templates and solution files.  File solutions are located 
in the `solutions` directory.  All template files are located in the
`scripts` directory.  Repo has the following dependencies:

1. matplotlib
2. numpy
3. os
4. geopy
5. mpl_toolkits.basemap
6. pandas

Run similar to


```
#!cmd
cd solutions
cd session-1
python solution-session-1.py
```


## Session 1

The first session plots the locations of 100 commercial buildings and their nearest weather
station.

![Alt Gray map of 100 commercial buildings and weather stations](https://bytebucket.org/blackmencode/bmc-core-data-vis/raw/f2e9ff12924d3e7455ee5f3b09773e8c54bfd6f3/figures/buildingslocs-session1.png)

## Session 2
The next session features the same commercial buildings, but this time the point locations of each
building are sized based on the annual energy use of the building.

![Alt White map of commercial buildings with points sized by energy use.](https://bytebucket.org/blackmencode/bmc-core-data-vis/raw/f2e9ff12924d3e7455ee5f3b09773e8c54bfd6f3/figures/buildingslocs-session2.png)

## Session 3
Final session gets away from the map plotting and doesn't depend on either `geopy` or `mpl_toolits.basemap`.  
Here we're plotting the relationship between annual energy use and conditioned floor area.  Points
are colored by building type.

![Alt Scatter plot of buildings energy use versus conditioned floor area.](https://bytebucket.org/blackmencode/bmc-core-data-vis/raw/f2e9ff12924d3e7455ee5f3b09773e8c54bfd6f3/figures/buildingsdata-session3.png)


Content originally developed by Michael Street of Black Men Code, Inc.