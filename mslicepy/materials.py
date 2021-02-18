"""Module :mod:`~mslicepy.materials` defines atomic and material
properties related to x-ray scattering, diffraction and propagation:
scattering factors, refractive index, absorption coefficient etc.
"""
import os
import numpy as np

ROOT_PATH = os.path.dirname(__file__)

ELEMENTS = ('None', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
            'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
            'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
            'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
            'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
            'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U')

class Element(object):
    """This class serves for accessing the scattering factors f1 and f2
    and atomic scattering factor of a chemical element `elem`.

    Parameters
    ----------
    elem : str or int
        The element can be specified by its name (case sensitive) or its
        atomic number.
    dbase : {'Henke', 'Chantler', 'BrCo'}, optional
        Database of the tabulated scattering factors of each element. The
        following keywords are allowed:

        * 'Henke' : (10 eV < E < 30 keV) [Henke]_.
        * 'Chantler' : (11 eV < E < 405 keV) [Chantler]_.
        * 'BrCo' : (30 eV < E < 509 keV) [BrCo]_.

    Attributes
    ----------
    name : str
        Name of the chemical element `elem`.
    atom_num : int
        Atomic number of `elem`.
    dbase : str
        Database of scattering factors.
    asf_coeffs : numpy.ndarray
        Coefficients of atomic scattering factor.
    sf_coeffs : numpy.ndarray
        Scattering factors (`energy`, `f1`, `f2`).
    mass : float
        Atomic mass [u].
    radius : float
        Atomic radius [Angstrom].
    density : float
        Density [g / cm^3].

    References
    ----------
    .. [Henke] http://henke.lbl.gov/optical_constants/asf.html
               B.L. Henke, E.M. Gullikson, and J.C. Davis, *X-ray interactions:
               photoabsorption, scattering, transmission, and reflection at
               E=50-30000 eV, Z=1-92*, Atomic Data and Nuclear Data Tables
               **54** (no.2) (1993) 181-342.
    .. [Chantler] http://physics.nist.gov/PhysRefData/FFast/Text/cover.html
                  http://physics.nist.gov/PhysRefData/FFast/html/form.html
                  C. T. Chantler, *Theoretical Form Factor, Attenuation, and
                  Scattering Tabulation for Z = 1 - 92 from E = 1 - 10 eV to E = 0.4 -
                  1.0 MeV*, J. Phys. Chem. Ref. Data **24** (1995) 71-643.
    .. [BrCo] http://www.bmsc.washington.edu/scatter/periodic-table.html
              ftp://ftpa.aps.anl.gov/pub/cross-section_codes/
              S. Brennan and P.L. Cowan, *A suite of programs for calculating
              x-ray absorption, reflection and diffraction performance for a
              variety of materials at arbitrary wavelengths*, Rev. Sci. Instrum.
              **63** (1992) 850-853.
    """
    dbase_lookup = {'Henke': 'data/henke_f1f2.npz', 'Chantler': 'data/chantler_f1f2.npz',
                    'BrCo': 'data/brco_f1f2.npz'}
    asf_dbase = 'data/henke_f0.npz'
    atom_dbase = 'data/atom_data.npz'

    def __init__(self, elem, dbase='Chantler'):
        if isinstance(elem, str):
            self.name = elem
            self.atom_num = ELEMENTS.index(elem)
        elif isinstance(elem, int):
            self.name = ELEMENTS[elem]
            self.atom_num = elem
        else:
            raise ValueError('Wrong element: {:s}'.format(str(elem)))
        if dbase in self.dbase_lookup:
            self.dbase = dbase
        else:
            raise ValueError('Wrong database: {:s}'.format(dbase))
        self._init_coeffs()

    def _init_coeffs(self):
        with np.load(os.path.join(ROOT_PATH, self.asf_dbase)) as dbase:
            self.asf_coeffs = dbase[self.name]
        with np.load(os.path.join(ROOT_PATH, self.dbase_lookup[self.dbase])) as dbase:
            self.sf_coeffs = np.stack((dbase[self.name + '_E'], dbase[self.name + '_f1'],
                                       dbase[self.name + '_f2']))
        with np.load(os.path.join(ROOT_PATH, self.atom_dbase)) as dbase:
            self.mass = dbase['mass'][self.atom_num]
            self.radius = dbase['radius'][self.atom_num]
            self.density = dbase['density'][self.atom_num]

    def get_asf(self, scat_vec):
        """Calculate atomic scattering factor for the given magnitude of
        scattering vector `scat_vec`.

        Parameters
        ----------
        scat_vec : float
            Scattering vector magnitude [Angstrom^-1].

        Returns
        -------
        asf : float
            Atomic scattering factor.
        """
        q_ofpi = scat_vec / 4 / np.pi
        asf = self.asf_coeffs[5] + sum(a * np.exp(-b * q_ofpi**2)
                                       for a, b in zip(self.asf_coeffs[:5], self.asf_coeffs[6:]))
        return asf

    def get_sf(self, energy):
        """Return a complex scattering factor (`f1` + 1j * `f2`) for the given photon
        `energy`.

        Parameters
        ----------
        energy : float or numpy.ndarray
            Photon energy [eV].

        Returns
        -------
        f1 + 1j * f2 : numpy.ndarray
            Complex scattering factor.
        """
        if np.any(energy < self.sf_coeffs[0, 0]) or np.any(energy > self.sf_coeffs[0, -1]):
            exc_txt = 'Energy is out of bounds: ({0:.2f} - {1:.2f})'.format(self.sf_coeffs[0, 0],
                                                                            self.sf_coeffs[0, -1])
            raise ValueError(exc_txt)
        f_one = np.interp(energy, self.sf_coeffs[0], self.sf_coeffs[1]) + self.atom_num
        f_two = np.interp(energy, self.sf_coeffs[0], self.sf_coeffs[2])
        return f_one + 1j * f_two

class Material:
    """
    :class:`Material` serves for getting refractive index and absorption
    coefficient of a material specified by its chemical formula and density.

    Parameters
    ----------
    elements : str or sequence of str
        List of all the constituent elements (symbols).
    quantities: None or sequence of float, optional
        Coefficients in the chemical formula. If None, the coefficients
        are all equal to 1.
    dbase : {'Henke', 'Chantler', 'BrCo'}, optional
        Database of the tabulated scattering factors of each element. The
        following keywords are allowed:

        * 'Henke' : (10 eV < E < 30 keV) [Henke]_.
        * 'Chantler' : (11 eV < E < 405 keV) [Chantler]_.
        * 'BrCo' : (30 eV < E < 509 keV) [BrCo]_.

    Attributes
    ----------
    elements : sequence of :class:`Element`
        List of elements.
    quantities : sequence of float
        Coefficients in the chemical formula.
    mass : float
        Molar mass [u].

    See Also
    --------
    Element - see for full description of the databases.
    """
    en_to_wl = 12398.419297617678 # h * c / e [eV * A]
    ref_cf = 2.7008646837561236e-06 # Avogadro * r_el / 2 / pi [A]

    def __init__(self, elements, quantities=None, dbase='Chantler'):
        if isinstance(elements, str):
            elements = [elements]
        if quantities is None:
            quantities = [1. for _ in elements]
        self.elements, self.quantities = [], []
        self.mass = 0
        for elem, quant in zip(elements, quantities):
            new_elem = Element(elem, dbase=dbase)
            self.elements.append(new_elem)
            self.quantities.append(quant)
            self.mass += new_elem.mass * quant

    def get_ref_index(self, energy, density):
        r"""Calculates the complex refractive index for the given photon
        `energy` and `density`.

        Parameters
        ----------
        energy : float or numpy.ndarray
            Photon energy [eV].
        density : float
            Physical density [g / cm^3].

        Returns
        -------
        mu : float or numpy.ndarray
            Linear absorption coefficient [cm^-1]

        Notes
        -----
        The complex refractive index is customary denoted as:
        .. math::
            n = 1 - \delta + i \betta
        The real and imaginary components, :math:`\delta` and :math:`\betta` are
        given by [GISAXS]_:
        .. math::
            \delta = \frac{\rho N_a r_e \lambda f_1}{2 \pi m_a}
        .. math::
            \betta = \frac{\rho N_a r_e \lambda f_2}{2 \pi m_a}
        where $\rho$ is physical density, $N_a$ is Avogadro constant, $m_a$ is
        atomic molar mass, $r_e$ is radius of electron, $lambda$ is wavelength,
        $f_1$ and f_2$ are real and imaginary components of scattering factor.

        Reference
        ---------
        .. [GISAXS] http://gisaxs.com/index.php/Refractive_index
        """
        wl = self.en_to_wl / energy
        ref_idx = np.sum([elem.get_sf(energy) * quant
                          for elem, quant in zip(self.elements, self.quantities)])
        return self.ref_cf * wl**2 * density * ref_idx / self.mass

    def get_absorption_coefficient(self, energy, density):
        r"""Calculates the linear absorption coefficientfor the given photon
        `energy` and `density`.

        Parameters
        ----------
        energy : float or numpy.ndarray
            Photon energy [eV].
        density : float
            Physical density [g / cm^3].

        Returns
        -------
        mu : float or numpy.ndarray
            Linear absorption coefficient [cm^-1]

        Notes
        -----
        The absorption coefficient is the inverse of the absorption length and
        is given by [GISAXS]_:
        .. math::
            \mu = \frac{\rho N_a}{m_a} 2 r_e \lambda f_2 = \frac{4 \pi \betta}{\lambda}
        where $\rho$ is physical density, $N_a$ is Avogadro constant, $m_a$ is
        atomic molar mass, $r_e$ is radius of electron, $lambda$ is wavelength,
        and $f_2$ is imaginary part of scattering factor.

        Reference
        ---------
        .. [GISAXS] http://gisaxs.com/index.php/Absorption_length
        """
        wl = self.en_to_wl / energy
        return 4 * np.pi * self.get_refractive_index(energy, density).imag / wl
