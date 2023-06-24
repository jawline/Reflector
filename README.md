# LAS to 3D Printer

This project produces 3D printable models from lidar data. Examples were
generated from freely available data from Hong Kong but this should work for
many lidar recordings taken aerially.

# Overview

The point cloud is sampled into a height map, then interpolated to fill in any
missing points before being output to a bin file for the viewer or final
conversion to an STL or a normalized png for any height mapping software.

# Instructions

First find some aerial lidar data in the LAS format. For my examples I sourced
my data from [here](https://www.geomap.cedd.gov.hk/GEOOpenData/eng/LIDAR.aspx).
Next, either view the data in the built in viewer for a preview or render it to
a heightmap or STL file using the instructions below.

## Creating a heightmap png for rendering in Blender

## Viewing LAS data in the viewer

## Creating an STL

TODO;

# Example printed models

<p float="left">
  <img src="/example prints/print1.jpg" width="80%" />
  <img src="/example prints/print2.jpg" width="80%" />
  <img src="/example prints/print3.jpg" width="80%" />
  <img src="/example prints/print4.jpg" width="80%" />
  <img src="/example prints/print5.jpg" width="80%" />
  <img src="/example prints/print6.jpg" width="80%" />
</p>

# Example renders

<p float="left">
  <img src="/example images/printable.png" width="80%" />
  <img src="/example images/plover_cove.png" width="80%" />
  <img src="/example images/central_1.png" width="80%" />
  <img src="/example images/central_2.png" width="80%" />
  <img src="/example images/central_3.png" width="80%" />
</p>
