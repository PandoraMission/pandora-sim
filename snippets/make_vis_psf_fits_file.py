"""Takes the LLNL matlab file and converts it into a fits file, for easier loading and lighter storage..."""
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from scipy.io import loadmat
import numpy as np
from datetime import datetime

# Vis PSF
#-----------------------------------#

d = loadmat('/Users/chedges/Downloads/pandora_vis_20220506.mat')
# Bin down the PSF, it's too big a file...
nbin = 2
PSF = np.asarray([[d['PSF'][idx::nbin, jdx::nbin] for idx in range(nbin)] for jdx in range(nbin)]).mean(axis=(0, 1))
PSF = PSF.reshape((*PSF.shape[:2], 9, 9, 5))
sub_pixel_size = nbin * d['dx'][0][0] * u.micron/u.pix
pixel_size = 6.5 * u.micron/u.pix
x = d['x'][0].reshape((9, 9, 5)) * u.mm
y = d['y'][0].reshape((9, 9, 5)) * u.mm
x = (x/pixel_size).to(u.pixel)
y = (y/pixel_size).to(u.pixel)
wvl = d['wvl'][0].reshape((9, 9, 5)) * u.micron

hdr = fits.Header([('AUTHOR1', 'LLNL'),
                   ('AUTHOR2', 'Christina Hedges'),
                   ("ORIGIN", "pandora_vis_20220506.mat"), 
                   ("CREATED", str(d['__header__']).split('Created on: ')[-1][:-1])
                   ("DATE", datetime.now().isoformat()),
                   ("PIXSIZE", pixel_size.value, f'PSF pixel size in {pixel_size.unit.to_string()}'),
                   ("SUBPIXSZ", sub_pixel_size.value, f'PSF sub pixel size in {sub_pixel_size.unit.to_string()}'),
                  ])
primaryhdu = fits.PrimaryHDU(header=hdr)
hdu = fits.HDUList([primaryhdu, fits.ImageHDU(PSF, name='PSF'),
              fits.ImageHDU(x.value, name='X'),
              fits.ImageHDU(y.value, name='Y'), 
              fits.ImageHDU(wvl.value, name='WAVELENGTH')])
hdu[2].header['UNIT'] = y.unit.to_string()
hdu[3].header['UNIT'] = x.unit.to_string()
hdu[4].header['UNIT'] = wvl.unit.to_string()
hdu.writeto('/Users/chedges/repos/pandora-sat/src/pandorasat/data/pandora_vis_20220506.fits', overwrite=True)


# NIR PSF
#-----------------------------------#

d = loadmat('/Users/chedges/Downloads/pandora_nir_20220506_PSF.mat')
# Bin down PSF, too big
nbin = 2
PSF = np.asarray([[d['PSF'][idx::nbin, jdx::nbin] for idx in range(nbin)] for jdx in range(nbin)]).mean(axis=(0, 1))
PSF = PSF.reshape((*PSF.shape[:2], 901))
sub_pixel_size = nbin * d['dx'][0][0] * u.micron/u.pix
pixel_size = 18 * u.micron/u.pix
# Bin down wavelength too, also too big.
nbin = 20
PSF = np.asarray([PSF[:, :, :900][:, :, idx::nbin] for idx in range(nbin)]).mean(axis=(0))
wvl = np.asarray([d['wvl'][:900][idx::nbin] for idx in range(nbin)]).mean(axis=(0))[:, 0] * u.micron

hdr = fits.Header([('AUTHOR1', 'LLNL'),
                   ('AUTHOR2', 'Christina Hedges'),
                   ("ORIGIN", "pandora_nir_20220506.mat"), 
                   ("CREATED", str(d['__header__']).split('Created on: ')[-1][:-1])
                   ("DATE", datetime.now().isoformat()),
                   ("PIXSIZE", pixel_size.value, f'PSF pixel size in {pixel_size.unit.to_string()}'),
                   ("SUBPIXSZ", sub_pixel_size.value, f'PSF sub pixel size in {sub_pixel_size.unit.to_string()}'),
                  ])
primaryhdu = fits.PrimaryHDU(header=hdr)
hdu = fits.HDUList([primaryhdu, fits.ImageHDU(PSF, name='PSF'),
              fits.ImageHDU(wvl.value, name='WAVELENGTH')])
hdu[2].header['UNIT'] = wvl.unit.to_string()
hdu.writeto('/Users/chedges/repos/pandora-sat/src/pandorasat/data/pandora_nir_20220506.fits', overwrite=True)

# PSF with thermal info
#-----------------------------------#

d = loadmat('/Users/chedges/Downloads/pandora_vis_20220506_hot_PSF_512.mat')
nbin = 2
PSF_hot = d['PSF'][:-1, :-1, :]
PSF_hot = np.asarray([[PSF_hot[idx::nbin, jdx::nbin] for idx in range(nbin)] for jdx in range(nbin)]).mean(axis=(0, 1))
PSF_hot = PSF_hot.reshape((*PSF_hot.shape[:2], 9, 9, 5))
d = loadmat('/Users/chedges/Downloads/pandora_vis_20220506_cold_PSF_512.mat')
nbin = 2
PSF_cold = d['PSF'][:-1, :-1, :]
PSF_cold = np.asarray([[PSF_cold[idx::nbin, jdx::nbin] for idx in range(nbin)] for jdx in range(nbin)]).mean(axis=(0, 1))
PSF_cold = PSF_cold.reshape((*PSF_cold.shape[:2], 9, 9, 5))

sub_pixel_size = nbin * d['dx'][0][0] * u.micron/u.pix
pixel_size = 6.5 * u.micron/u.pix
x = d['x'][0].reshape((9, 9, 5)) * u.mm
y = d['y'][0].reshape((9, 9, 5)) * u.mm
x = (x/pixel_size).to(u.pixel)
y = (y/pixel_size).to(u.pixel)
wvl = d['wvl'][0].reshape((9, 9, 5)) * u.micron
temp = np.asarray([-10., 30.])*u.deg_C
PSF = np.asarray([PSF_cold, PSF_hot]).transpose([1, 2, 3, 4, 5, 0])
x = x[:, :, :, None] * np.ones(len(temp))
y = y[:, :, :, None] * np.ones(len(temp))
wvl = wvl[:, :, :, None] * np.ones(len(temp))
temp = temp[None, None, None, :] * np.ones(wvl.shape)

hdr = fits.Header([('AUTHOR1', 'LLNL'),
                   ('AUTHOR2', 'Christina Hedges'),
                   ("ORIGIN1", "pandora_vis_20220506_cold_PSF_512.mat"), 
                   ("ORIGIN2", "pandora_vis_20220506_hot_PSF_512.mat"), 
                   ("CREATED", str(d['__header__']).split('Created on: ')[-1][:-1]),
                   ("DATE", datetime.now().isoformat()),
                   ("PIXSIZE", pixel_size.value, f'PSF pixel size in {pixel_size.unit.to_string()}'),
                   ("SUBPIXSZ", sub_pixel_size.value, f'PSF sub pixel size in {sub_pixel_size.unit.to_string()}'),
                  ])
primaryhdu = fits.PrimaryHDU(header=hdr)
hdu = fits.HDUList([primaryhdu, fits.ImageHDU(PSF, name='PSF'),
              fits.ImageHDU(x.value, name='X'),
              fits.ImageHDU(y.value, name='Y'), 
              fits.ImageHDU(wvl.value, name='WAVELENGTH'),
              fits.ImageHDU(temp.value, name='TEMPERATURE')])
hdu[2].header['UNIT'] = y.unit.to_string()
hdu[3].header['UNIT'] = x.unit.to_string()
hdu[4].header['UNIT'] = wvl.unit.to_string()
hdu[5].header['UNIT'] = temp.unit.to_string()
hdu.writeto('/Users/chedges/repos/pandora-sat/src/pandorasat/data/pandora_vis_20220506.fits', overwrite=True)