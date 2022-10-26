"""Takes the LLNL matlab file and converts it into a fits file, for easier loading and lighter storage..."""
from astropy.io import fits
from astropy.table import Table
d = loadmat('/Users/chedges/Downloads/pandora_vis_20220506.mat')
# Bin down the PSF, it's too big a file...
nbin = 2
PSF = np.asarray([[d['PSF'][idx::nbin, jdx::nbin] for idx in range(nbin)] for jdx in range(nbin)]).mean(axis=(0, 1))
PSF = PSF.reshape((*PSF.shape[:2], 9, 9, 5))
sub_pixel_size = nbin * d['dx'][0][0] * u.micron/u.pix
pixel_size = 6.5 * u.micron/u.pix
x = d['x'][0].reshape((9, 9, 5)) * u.mm
y = d['y'][0].reshape((9, 9, 5)) * u.mm
wvl = d['wvl'][0].reshape((9, 9, 5)) * u.micron

hdr = fits.Header([('AUTHOR', 'Christina Hedges'),
                   ("ORIGIN", "pandora_vis_20220506.mat"), 
                   ("DATE", "Wed Jul 20 12:18:40 2022"),
                   ("PIXSIZE", pixel_size.value, f'PSF pixel size in {pixel_size.unit.to_string()}'),
                   ("SUBPIXSZ", sub_pixel_size.value, f'PSF sub pixel size in {sub_pixel_size.unit.to_string()}'),
                  ])
primaryhdu = fits.PrimaryHDU(header=hdr)
hdu = fits.HDUList([primaryhdu, fits.ImageHDU(PSF, name='PSF'),
              fits.ImageHDU(wvl.value, name='WAVELENGTH'),
              fits.ImageHDU(x.value, name='X'),
              fits.ImageHDU(y.value, name='Y')])
hdu[2].header['UNIT'] = wvl.unit.to_string()
hdu[3].header['UNIT'] = x.unit.to_string()
hdu[4].header['UNIT'] = y.unit.to_string()
hdu.writeto('pandora_vis_20220506.fits', overwrite=True)