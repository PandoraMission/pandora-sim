import pandorasim as psim
import pandorapsf as pp
import pandorasat as ps
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from pandorasim.utils import get_jitter
from astropy.coordinates import SkyCoord
import astropy.units as u


def get_jitter_test_cube_FFI(
    targetname="WASP-69",
    nreads: int = 50,
    jitter_row: float = 1,
    jitter_col: float = 1,
    noise=True,
    jitter=True,
    cosmic_ray_rate=1000 / (u.second * u.cm**2),
):
    """Get a fits file with reads of the visible channel FFIs with an expected amount of jitter."""

    c = SkyCoord.from_name(targetname)
    # Initialize a simulator object
    sim = psim.VisibleSim()
    # Point the simulator
    sim.point(c.ra, c.dec, -40 * u.deg)

    int_time = sim.detector.integration_time
    source_flux = (
        (np.asarray(sim.source_catalog.counts) * u.electron / u.second) * int_time
    ).value.astype(int)[:, None] * np.ones(nreads)

    if jitter:
        _, row, column, _ = get_jitter(
            rowstd=(u.Quantity(jitter_row, u.arcsecond) / sim.detector.pixel_scale)
            .to(u.pixel)
            .value,
            colstd=(u.Quantity(jitter_col, u.arcsecond) / sim.detector.pixel_scale)
            .to(u.pixel)
            .value,
            nframes=nreads,
            frame_time=int_time.to(u.second).value,
        )
        delta_pos = np.asarray([row, column])
    else:
        delta_pos = None
    # FFI has shape (nrows, ncolumns), in units of electrons.
    ffis = sim.scene.model(source_flux, delta_pos)
    # Apply poisson (shot) noise, ffi now has shape  (nrows, ncolumns), units of electrons
    # if hasattr(sim.detector, "fieldstop"):
    #     ffis *= sim.detector.fieldstop.astype(int)

    if noise:
        # Apply background to every read, units of electrons
        bkg = np.random.poisson(
            (sim.background_rate * int_time).value, size=ffis.shape
        ).astype(int)
        # if hasattr(sim.detector, "fieldstop"):
        #     bkg *= sim.detector.fieldstop.astype(int)
        ffis += bkg

        # # Apply a bias to every read which is a Gaussian with mean = bias * nreads value and std = (nreads * (read noise)**2)**0.5
        # We actually do this as a sum because otherwise the integer math doesn't work out...!?

        test_distribution = (
            np.random.normal(
                loc=sim.detector.bias.value,
                scale=sim.detector.read_noise.value,
                size=(nreads, 10000),
            )
            .astype(int)
            .sum(axis=0)
        )
        ffis += np.random.normal(
            loc=test_distribution.mean(),
            scale=sim.detector.read_noise.value * np.sqrt(nreads),
            size=(ffis.shape),
        ).astype(int)

        # Add poisson noise for the dark current to every frame, units of electrons
        ffis += np.random.poisson(
            lam=(sim.detector.dark_rate * int_time).value,
            size=ffis.shape,
        ).astype(int)

    cosmic_ray_expectation = (
        cosmic_ray_rate
        * ((sim.detector.pixel_size * 2048 * u.pix) ** 2).to(u.cm**2)
        * sim.detector.integration_time
    ).value
    for idx in range(nreads):
        ffis[idx] += psim.utils.get_simple_cosmic_ray_image(
            cosmic_ray_expectation=cosmic_ray_expectation
        ).value

    # Apply gain
    #        ffi = sim.detector.apply_gain(u.Quantity(ffi.ravel(), unit='electron')).value.reshape(ffi.shape)
    # Crap gain for now because gain calculations are wicked broken
    # ffi *= sim.detector.gain

    # This is a bit hacky, but for FFIs we'll be ok. We do this because working with individual reads for FFI data is slow.
    ffis[ffis > (nreads * 2**16)] = nreads * 2**16

    hdr = fits.Header()
    hdr["AUTHOR"] = "Christina Hedges"
    hdr["VERSION"] = psim.__version__
    hdr["DATE"] = Time.now().strftime("%d-%m-%Y")
    hdr["TARGET"] = targetname
    hdr["RA_OBJ"] = c.ra.deg
    hdr["DEC_OBJ"] = c.dec.deg
    hdr["COSMICS"] = (True, "are cosmic rays simulated in this dataset")
    hdr["CSMCRATE"] = (
        cosmic_ray_rate.value,
        "rate of cosmic rays in number/second/cm^2",
    )
    hdr["DARK"] = (sim.detector.dark_rate.value, "dark current in electrons/second")
    hdr["READNSE"] = (sim.detector.read_noise.value, "read noise in electrons")
    hdr["INT_TIME"] = (
        sim.detector.integration_time.value,
        "integration time in seconds",
    )
    hdr["JITTER_T"] = (0.0005, "jitter correlation timescale")
    hdr["JITTER_X"] = (jitter_col, "jitter stddev in x in arcseconds")
    hdr["JITTER_Y"] = (jitter_row, "jitter stddev in y in arcseconds")
    hdr["JITTER_R"] = (0.0005, "jitter stddev in theta")
    hdu0 = fits.PrimaryHDU(header=hdr)
    cols = fits.ColDefs(
        [
            fits.Column(
                name="time",
                array=np.arange(nreads) * int_time,
                unit="second",
                format="D",
            ),
            fits.Column(name="xcenter", array=delta_pos[1], unit="pixel", format="D"),
            fits.Column(name="ycenter", array=delta_pos[0], unit="pixel", format="D"),
        ]
    )
    # fits.Column(name='rotation', array=rotation.value + startrotation.value, unit='pixel', format='D'),])
    hdulist = fits.HDUList(
        [
            hdu0,
            fits.BinTableHDU.from_columns(cols, name="META"),
            fits.BinTableHDU(Table.from_pandas(sim.source_catalog), name="sources"),
            fits.CompImageHDU(data=np.int32(ffis), name="science"),
        ]
    )
    return hdulist


def main():
    targets = [
        "WASP-69",
        "WASP-107",
        "HIP 65 A",
        "TOI-3884",
        "GJ 1214",
        "WASP-177",
        "WASP-80",
        "WASP-52",
        "TOI-942",
        "K2-198",
        "L 98-59",
        "TOI-2427",
        "TOI-168",
        "TOI-836",
        "TOI-1416",
        "HD 3167",
        "TOI-776",
        "GJ 9827",
        "TOI-244",
        "LTT 1445 A",
    ]
    for target in targets:
        for cmrate in np.arange(1, 4):
            hdulist = get_jitter_test_cube_FFI(
                targetname=target, cosmic_ray_rate=10**cmrate * 1 / (u.second * u.cm**2)
            )
            hdulist.writeto(
                f"calibrationtests/jittercubes/{target.replace(' ', '_')}_cmrate{cmrate}_visible_sim_{Time.now().strftime('%d-%m-%Y')}.fits",
                overwrite=True,
                checksum=True,
            )
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(
                hdulist[3].data.sum(axis=0) - hdulist[3].data.sum(axis=0).min(),
                origin="lower",
                vmin=0,
                vmax=1500,
                cmap="Greys_r",
            )
            ax.set(
                xlabel="Column [pixel]",
                ylabel="Row [pixel]",
                title=hdulist[0].header["TARGET"],
            )
            fig.savefig(
                f"calibrationtests/jittercubes/{target.replace(' ', '_')}_cmrate{cmrate}_visible_sim_{Time.now().strftime('%d-%m-%Y')}.png",
                dpi=200,
                bbox_inches="tight",
            )


if __name__ == "__main__":
    main()
