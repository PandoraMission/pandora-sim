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
import matplotlib.animation as animation


def get_jitter_test_cube_FFI(
    targetname="WASP-69",
    nreads: int = 50,
    jitter=True,
    cosmic_ray_rate=1000 / (u.second * u.cm**2),
):
    """Get a fits file with reads of the visible channel FFIs with an expected amount of jitter."""

    c = SkyCoord.from_name(targetname)
    c = ps.utils.get_sky_catalog(c.ra.deg, c.dec.deg, radius=0.015)["coords"][0]

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
            rowstd=(u.Quantity(3, u.arcsecond) / sim.detector.pixel_scale)
            .to(u.pixel)
            .value,
            colstd=(u.Quantity(3, u.arcsecond) / sim.detector.pixel_scale)
            .to(u.pixel)
            .value,
            correlation_time=1.5 * u.second,
            nframes=nreads,
            frame_time=int_time.to(u.second).value,
        )
        delta_pos = np.asarray([row, column])
    else:
        delta_pos = None
    # FFI has shape (nrows, ncolumns), in units of electrons.
    ffis = sim.scene.model(source_flux, delta_pos)
    k = ffis > 0
    ffis[k] = np.random.poisson(ffis[k])

    gain = 1 / 0.6
    # Convert to DN, hand coded
    ffis *= gain

    cosmic_ray_expectation = (
        cosmic_ray_rate
        * ((sim.detector.pixel_size * 2048 * u.pix) ** 2).to(u.cm**2)
        * sim.detector.integration_time
    ).value

    cmrs = np.zeros_like(ffis)
    for idx in range(nreads):
        cmr = psim.utils.get_simple_cosmic_ray_image(
            cosmic_ray_expectation=cosmic_ray_expectation
        ).value
        k = cmr > 0
        # Gain hand coded
        cmrs[idx, k] += np.random.poisson(cmr[k]) * gain

    # Apply background to every read, units of electrons
    noise = np.random.poisson(
        (sim.background_rate * int_time).value, size=ffis.shape
    ).astype(int)

    # Add poisson noise for the dark current to every frame, units of electrons
    noise += np.random.poisson(
        lam=(sim.detector.dark_rate * int_time).value,
        size=ffis.shape,
    ).astype(int)

    # Apply a bias and read noise
    noise += np.random.normal(
        loc=0,
        scale=sim.detector.read_noise.value,
        size=(ffis.shape),
    ).astype(int)

    noise = np.asarray(np.asarray(noise, float) * gain, int)
    noise += sim.detector.bias.value.astype(int)
    ffis[ffis > (43000)] = 43000

    hdr = fits.Header()
    hdr["AUTHOR"] = "Christina Hedges"
    hdr["VERSION"] = psim.__version__
    hdr["DATE"] = Time.now().strftime("%d-%m-%Y")
    hdr["TARGET"] = targetname
    hdr["RA_OBJ"] = (c.ra.deg, "RA of the object")
    hdr["DEC_OBJ"] = (c.dec.deg, "Dec of the object")
    hdr["ROLL"] = (-40, "roll angle in degrees")
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
    hdr["JITTER_T"] = (1.5, "jitter correlation timescale [s]")
    hdr["JITTER_X"] = (3, "jitter stddev in x in arcseconds")
    hdr["JITTER_Y"] = (3, "jitter stddev in y in arcseconds")
    hdr["SCI_UNIT"] = ("DN", "unit of science extension")
    hdu0 = fits.PrimaryHDU(header=hdr)
    ra_t, dec_t = sim.pixel_to_world(
        row=(delta_pos[0] + sim.wcs.wcs.crpix[1]),
        column=(delta_pos[1] + sim.wcs.wcs.crpix[0]),
        type="wcs",
    )
    roll_t = np.ones_like(ra_t) * -40.0
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
            fits.Column(name="RA_center", array=ra_t, unit="pixel", format="D"),
            fits.Column(name="Dec_center", array=dec_t, unit="pixel", format="D"),
            fits.Column(name="Roll", array=roll_t, unit="pixel", format="D"),
        ]
    )
    # fits.Column(name='rotation', array=rotation.value + startrotation.value, unit='pixel', format='D'),])
    hdulist = fits.HDUList(
        [
            hdu0,
            fits.BinTableHDU.from_columns(cols, name="META"),
            fits.BinTableHDU(Table.from_pandas(sim.source_catalog), name="sources"),
            fits.CompImageHDU(data=np.int32(ffis + noise + cmrs), name="science"),
            # fits.CompImageHDU(data=np.int32(ffis), name="photons"),
            # fits.CompImageHDU(data=np.int32(cmrs), name="cosmics"),
        ]
    )
    for card in sim.wcs.to_header(relax=True).cards:
        hdulist[3].header[card[0]] = (card[1], card[2])
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

    targets = [
        "WASP-69",
        "WASP-107",
        "HIP 65 A",
        "TOI-540",
        "K2-141",
        "K2-3",
        "GJ 1132",
        "GJ 3470",
        "GJ 357",
        "GJ 436",
        "GJ 9827",
        "HAT-P-11",
        "HAT-P-19",
        "HATS-72",
        "HIP 65 A",
        "L 98-59",
        "LHS_3844",
        "LTT 1445 A",
        "TRAPPIST-1",
    ]

    for target in targets:
        for cmrate in np.arange(2, 3):
            hdulist = get_jitter_test_cube_FFI(
                targetname=target,
                cosmic_ray_rate=10**cmrate * 1 / (u.second * u.cm**2),
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
