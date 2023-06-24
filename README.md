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

Run `cargo run --release --bin converter -- --las-folder-path <PATH TO FOLDER CONTAINING LAS DATA> --pixels-per-unit-dim <use configured resolution per unit (x, _, z) square in the las file. 1 is a reasonable default> --rounds-of-interpolated-hole-filling 100 --consider-nearest-n-neighbors-for-interpolation 8 --output-path /tmp/test.png --base-depth 60`.

## Viewing LAS data in the viewer

First run `cargo run --release --bin converter -- --las-folder-path <PATH TO FOLDER CONTAINING LAS DATA> --pixels-per-unit-dim <use configured resolution per unit (x, _, z) square in the las file. 1 is a reasonable default> --rounds-of-interpolated-hole-filling 100 --consider-nearest-n-neighbors-for-interpolation 8 --write-to-bin --output-path /tmp/test.bin --base-depth 60`
to convert the LAS data to a bin file for viewing Next run `RUST_LOG=viewer=info cargo run --release --bin viewer -- /tmp/test.bin`.

## Creating an STL

Run `cargo run --release --bin converter -- --las-folder-path <PATH TO FOLDER CONTAINING LAS DATA> --pixels-per-unit-dim 1 --rounds-of-interpolated-hole-filling 100 --consider-nearest-n-neighbors-for-interpolation 2 --write-to-stl  --output-path /tmp/test.stl --base-depth 60` to create /tmp/test.stl.

# Example printed models

<p float="left">
  <img src="/example prints/print1.jpg" width="30%" />
  <img src="/example prints/print2.jpg" width="30%" />
  <img src="/example prints/print3.jpg" width="30%" />
  <img src="/example prints/print4.jpg" width="30%" />
  <img src="/example prints/print5.jpg" width="30%" />
  <img src="/example prints/print6.jpg" width="30%" />
</p>

# Example renders

<p float="left">
  <img src="/example images/printable.png" width="30%" />
  <img src="/example images/plover_cove.png" width="30%" />
  <img src="/example images/central_1.png" width="30%" />
  <img src="/example images/central_2.png" width="30%" />
  <img src="/example images/central_3.png" width="30%" />
</p>
