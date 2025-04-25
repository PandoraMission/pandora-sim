# WCS Conventions

WCS conventions are thorny, this notebook walks through some of the pitfalls.


```python
import pandorasim as psim
import pandorasat as ps
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
```

## WCS Column-major indexing

In Python and numpy we have Row-major indexing. When we look at the shape of a 2D array the first element is the row. When we index into the array we will obtain the row values. 

WCS is Column major. This means that attributes such as CRPIX are given in the order 

```
    (x, y)
```
which we interpret as
```
    (column, row)
```

Functions such as `wcs.all_world2pix` return `(x, y)`, i.e. `(column, row)`

It is vital that we do not get these confused. 

## WCS Origins

WCS are written with a built in standard that the origin of the transformation is from the **middle** of the **first** pixel

```
--------------
|            |
|            |
|      .     |
|            |
|            |
--------------
```

What we label this origin depends on whether our coding language is 0 indexed or 1 indexed. WCS is 1 indexed. In Python we use 0 based indexing. However, in Python we would treat the origin as the lower left corner of the image, and the center of the first pixel as (0.5, 0.5). In WCS we treat the origin as the middle of the first pixel.

```
    WCS                      Python

--------------            --------------
|            |            |            |
|            |            |            |
|      .     |            |      .     |
|   (1, 1)   |            | (0.5, 0.5) |
|            |            |            |
--------------            --------------
```

Here are the key statistics for each indexing base

|                                  | 0 Index [Python] | 1 Index [Fortran]| WCS          |
| -------------------------------- | ---------------- | ---------------- | ------------ |
| Detector Size                    | (2048, 2048)     | (2048, 2048)     | (2048, 2048) |
| Lower Left Corner of First Pixel | (0, 0)           | (1, 1)           | (0.5, 0.5)   |
| Center of First pixel            | (0.5, 0.5)       | (1.5, 1.5)       | (1, 1)       |
| Center of Detector               | (1023.5, 1023.5) | (1024.5, 1024.5) | (1024, 1024) |

So for WCS, where we assume the origin of `1` corresponds to the center of the first pixel, (1024, 1024) is the center of the detector. 

WCS uses CRVAL and CRPIX to set the origin of the transformation. 

- CRVAL is the RA and Dec value at CRPIX
- CRPIX is the Column and Row value at CRVAL

## Examples

We can demonstrate that we get the right answer when we translate between these two using an origin of 1, i.e. 1-based indexing. Let's use the WCS for the visible detector


```python
c = SkyCoord.from_name("HD209458")
c = ps.utils.get_sky_catalog(c.ra.deg, c.dec.deg, radius=0.015)["coords"][0]
visda = ps.VisibleDetector()
wcs = visda.get_wcs(ra=c.ra, dec=c.dec, theta=-40*u.deg)
```


```python
wcs
```




    WCS Keywords
    
    Number of WCS axes: 2
    CTYPE : 'RA---TAN-SIP' 'DEC--TAN-SIP' 
    CRVAL : 330.79510746248036 18.88419298633337 
    CRPIX : 1024.0 1024.0 
    PC1_1 PC1_2  : 0.766044443118978 0.6427876096865393 
    PC2_1 PC2_2  : -0.6427876096865393 0.766044443118978 
    CDELT : -0.00021666666666666 0.000216666666666666 
    NAXIS : 2048.0  2048.0




```python
# Convert pixels to RA/Dec
# Use the origin=1 for 1 based indexing
ra, dec = wcs.all_pix2world(*wcs.wcs.crpix, 1)

# Should match CRVAL
assert np.isclose(ra, wcs.wcs.crval[0], rtol=1e-8)
assert np.isclose(dec, wcs.wcs.crval[1], rtol=1e-8)
```


```python
# Convert RA/Dec to pixels
# Use the origin=1 for 1 based indexing
column, row = wcs.all_world2pix(*wcs.wcs.crval, 1)

# Should match CRPIX
assert np.isclose(column, wcs.wcs.crpix[0], atol=1e-5)
assert np.isclose(row, wcs.wcs.crpix[1], atol=1e-5)
```

This works well. We need to be careful in general that we use 1 based indexing when we're comparing parts of the WCS. However when we state the postion of targets, we want to use Python indexing. In Python, if we wanted to find the pixel that the target lands on we would use


```python
column, row = wcs.all_world2pix(c.ra.deg, c.dec.deg, 0)
column, row
```




    (array(1023.), array(1022.99984865))



However, in Python we would treat the lower left corner as the origin and not the center. To account for this we must add half a pixel. 


```python
column, row = wcs.all_world2pix(c.ra.deg, c.dec.deg, 0)
column += 0.5
row += 0.5
```


```python
column, row
```




    (array(1023.5), array(1023.49984865))



This is now the position that the peak of the star flux would land on the detector. 

In the `pandorasim` tool there is a convenience function to help you convert between pixels and world coordinate space, accounting for these issues. 


```python
sim = psim.VisibleSim()
sim.point(c.ra, c.dec, -40 * u.deg)
```


```python
sim.wcs
```




    WCS Keywords
    
    Number of WCS axes: 2
    CTYPE : 'RA---TAN-SIP' 'DEC--TAN-SIP' 
    CRVAL : 330.79510746248036 18.88419298633337 
    CRPIX : 1024.0 1024.0 
    PC1_1 PC1_2  : 0.766044443118978 0.6427876096865393 
    PC2_1 PC2_2  : -0.6427876096865393 0.766044443118978 
    CDELT : -0.00021666666666666 0.000216666666666666 
    NAXIS : 2048.0  2048.0



You can convert RA and Dec to row and column. When you specify `"python"` as the type you will obtain the position in 0 indexed pixel coordinates, where we assume the lower left corner of the first pixel is the origin.


```python
sim.world_to_pixel(ra=c.ra, dec=c.dec, type='python')
```




$[1023.4998,~1023.5] \; \mathrm{pix}$



When you specify `"wcs"` as the type you will obtain the position in 1 indexed pixel coordinates, where we assume the middle of the first pixel is the origin.


```python
sim.world_to_pixel(ra=c.ra, dec=c.dec, type='wcs')
```




$[1023.9998,~1024] \; \mathrm{pix}$



There is an equivalent `pixel_to_world` function which will convert between pixels and RA and Dec. Note here we assume when you select `type="python"` you are putting in 0 indexed pixel coordinates where the origin is the lower left corner of the first pixel.


```python
sim.pixel_to_world(row=1023.5, column=1023.5, type='python')
```




$[330.79511,~18.884193] \; \mathrm{{}^{\circ}}$



Equivalently we can do this with the `wcs` type, where we assume 1 indexing and the middle of the first pixel is the origin


```python
sim.pixel_to_world(row=1024, column=1024, type='wcs')
```




$[330.79511,~18.884193] \; \mathrm{{}^{\circ}}$



We can demonstate these work by showing the round trip produces the same as the input

Here is the input


```python
np.asarray([c.ra.value, c.dec.value]) * u.deg
```




$[330.79511,~18.884193] \; \mathrm{{}^{\circ}}$



And then the round trip


```python
# round trip in `python`
sim.pixel_to_world(*(sim.world_to_pixel(ra=c.ra, dec=c.dec, type='python')), type='python')
```




$[330.79511,~18.884193] \; \mathrm{{}^{\circ}}$




```python
# round trip in `wcs`
sim.pixel_to_world(*sim.world_to_pixel(ra=c.ra, dec=c.dec, type='wcs'), type='wcs')
```




$[330.79511,~18.884193] \; \mathrm{{}^{\circ}}$


