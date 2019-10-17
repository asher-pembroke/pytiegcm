This project is used for building interpolators and analysis tools for the TIEGCM model.

## Requirements
	
* netCDF4
* numpy
* pandas
* scipy

## Changlog

Below are Summer 2019 changes provided by Zach Waldron

* Added helium to the diffusive equilibrium extrapolation above the boundary
* Changed the top boundary to 25th pressure level to avoid errors from numerical issues (issue with Tiegcm2.0)
* Changed any use of 'Z' for the vertical coordinate to 'ZG'
* Any points that exists on the midpoint levels use 'ZGMID' as vertical coordinate.
* If code is using ZGMID, extrapolate from top ZGMID boundary
* Changed the way that mass mixing ratio is converted to mass density for each variable
* Changed code to be more general to different names for N2 (CCMC runs on request named it N2N for some reason)
* Made the code more general/robust so that it can accept Both regular output types (from Cheyenne/NCAR) and the CCMC types from Runs on Request.