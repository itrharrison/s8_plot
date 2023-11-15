import pyccl as ccl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

plt.close('all') # tidy up any unshown plots

from scipy import constants

oneover_chmpc = 1. / (constants.c * 1.e5)

def smail_distribution(z, zm, alpha, beta, gamma):

    z0 = zm / alpha

    return (z**beta) * np.exp(-(z/z0)**gamma)


def interpolate_nz(filename, zs):

    z, nz = np.loadtxt(filename, delimiter=',', unpack=True)

    nz /= np.trapz(nz, z)

    return np.interp(zs, z, nz, left=0, right=0)

cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05,
                          h=0.7, n_s=0.95, sigma8=0.8,
                          transfer_function='bbks')

samples = {}

# z_gals = np.linspace(0.00001, 3., 2048)
z_gals = np.logspace(-5, np.log10(3.), 2048)
# z_cmbk = np.linspace(0.00001, 1100, 2048)
z_cmbk = np.logspace(-5, np.log10(1100), 2048)
cmbk = ccl.CMBLensingTracer(cosmo, z_source=1100)

nz_unwiseblue = interpolate_nz('data/unwise_blue.txt', z_gals)
unwiseblue = ccl.NumberCountsTracer(cosmo, dndz=(z_gals, nz_unwiseblue), bias=(z_gals, 0.8 + 1.2*z_gals - 1), has_rsd=False)
# unwiseblue = ccl.NumberCountsTracer(cosmo, dndz=(z_gals, nz_unwiseblue), bias=(z_gals, np.ones_like(z_gals)), has_rsd=False)

nz_unwisegreen = interpolate_nz('data/unwise_green.txt', z_gals)
unwisegreen = ccl.NumberCountsTracer(cosmo, dndz=(z_gals, nz_unwisegreen), bias=(z_gals, [max(1.6 * z**2., 1.) - 1 for z in z_gals]), has_rsd=False)
# unwisegreen = ccl.NumberCountsTracer(cosmo, dndz=(z_gals, nz_unwisegreen), bias=(z_gals, np.ones_like(z_gals)), has_rsd=False)

nz_desi_lrg = interpolate_nz('data/desi_lrg.txt', z_gals)
desilrg = ccl.NumberCountsTracer(cosmo, dndz=(z_gals, nz_desi_lrg), bias=(z_gals, [max(1.6 * z**2., 1.) - 1 for z in z_gals]), has_rsd=False)

nz_desy3 = interpolate_nz('data/des-y3_sources.txt', z_gals)
nz_desy3[-100:] = 0.0
desy3 = ccl.WeakLensingTracer(cosmo, (z_gals, nz_desy3))

nz_kids1000 = interpolate_nz('data/kids1000_sources.txt', z_gals)
kids1000 = ccl.WeakLensingTracer(cosmo, (z_gals, nz_kids1000))

nz_hsc = interpolate_nz('data/hsc_sources.txt', z_gals)
hsc = ccl.WeakLensingTracer(cosmo, (z_gals, nz_hsc))

samples['CMB Lensing'] = cmbk
samples['DES-Y3 Sources'] = desy3
samples['KiDS-1000'] = kids1000
samples['HSC'] = hsc
samples['unWISE Blue gals'] = unwiseblue
samples['unWISE Green gals'] = unwisegreen
samples['DESI LRGs'] = desilrg

chi_max = cosmo.comoving_radial_distance(1 / (1. + z_gals[-1]))
chi_grid = np.linspace(0.0, chi_max, z_gals.shape[0])

chiatz = cosmo.comoving_radial_distance(1 / (1. + z_gals))
zatchi = np.interp(chi_grid, chiatz, z_gals)

w_1 = desilrg.get_kernel(chi_grid)[0]
w_2 = desilrg.get_kernel(chi_grid)[0]

kern = w_1 * w_2 / chi_grid**2.

zeff = np.trapz(kern[1:] * zatchi[1:], x=chi_grid[1:]) / np.trapz(kern[1:], x=chi_grid[1:])

print(zeff)


nz_des = smail_distribution(z_cmbk, 0.6, np.sqrt(2), 2., 1.5)
desg = ccl.WeakLensingTracer(cosmo, (z_cmbk, nz_des))

nz_skao1 = smail_distribution(z_cmbk, 1.1, np.sqrt(2), 2., 1.25)
skao1g = ccl.WeakLensingTracer(cosmo, (z_cmbk, nz_skao1))

# plot kernels

def get_kernel(t_1, cosmo, zs):

    chis = cosmo.comoving_radial_distance(1 / (1. + zs))

    chi_grid = np.linspace(0.0001, chi_max, zs.shape[0])
    chiatz = cosmo.comoving_radial_distance(1 / (1. + zs))
    zatchi = np.interp(chi_grid, chiatz, zs)

    Hz = cosmo.h_over_h0(1. / (1. + zs)) * cosmo['h'] * 100.
    Dchi = cosmo.growth_factor(1. / (1. + zatchi))

    w_1 = t_1.get_kernel(chis)[0] / Hz

    w_1 /= w_1.max()

    return w_1


def get_zeff(t_1, t_2, cosmo, zs):

    chis = cosmo.comoving_radial_distance(1 / (1. + zs))

    chi_grid = np.linspace(0.0001, chi_max, zs.shape[0])
    chiatz = cosmo.comoving_radial_distance(1 / (1. + zs))
    zatchi = np.interp(chi_grid, chiatz, zs)

    Hz = cosmo.h_over_h0(1. / (1. + zatchi)) * cosmo['h'] * 100.
    Dchi = cosmo.growth_factor(1. / (1. + zatchi))

    w_1 = t_1.get_kernel(chi_grid)[0]# * Dchi
    w_2 = t_2.get_kernel(chi_grid)[0]# * Dchi

    num_integrand = w_1 * w_2 * zatchi / chi_grid**2
    den_integrand = w_1 * w_2 / chi_grid**2

    zeff = np.trapz(num_integrand[1:], x=chi_grid[1:]) / np.trapz(den_integrand[1:], x=chi_grid[1:])

    return zeff

zeff_cmbk = get_zeff(cmbk, cmbk, cosmo, z_cmbk)
zeff_desg = get_zeff(desg, desg, cosmo, z_gals)
zeff_skao1g = get_zeff(skao1g, skao1g, cosmo, z_gals)

zeff_cmbkxdesg = get_zeff(cmbk, desg, cosmo, z_cmbk)
zeff_cmbkxskao1f = get_zeff(cmbk, skao1g, cosmo, z_cmbk)

zeff_unwiseblue = get_zeff(unwiseblue, unwiseblue, cosmo, z_gals)
zeff_unwisegreen = get_zeff(unwisegreen, unwisegreen, cosmo, z_gals)
zeff_desilrg = get_zeff(desilrg, desilrg, cosmo, z_gals)

zeff_cmbkxunwiseblue = get_zeff(cmbk, unwiseblue, cosmo, z_cmbk)
zeff_cmbkxunwisegreen = get_zeff(cmbk, unwisegreen, cosmo, z_cmbk)
zeff_cmbkxdesilrg = get_zeff(cmbk, desilrg, cosmo, z_cmbk)

print('des-y3: {}'.format(get_zeff(desy3, desy3, cosmo, z_cmbk)))
print('CMB lensing x des-y3: {}'.format(get_zeff(cmbk, desy3, cosmo, z_cmbk)))
print('kids1000: {}'.format(get_zeff(kids1000, kids1000, cosmo, z_cmbk)))
print('hsc: {}'.format(get_zeff(hsc, hsc, cosmo, z_cmbk)))
print('skao1: {}'.format(get_zeff(skao1g, skao1g, cosmo, z_cmbk)))

print('cmb lensing: {}'.format(zeff_cmbk))

print('unwiseblue: {}'.format(zeff_unwiseblue))
print('unwisegreen: {}'.format(zeff_unwisegreen))

print('CMB lensing x unwiseblue: {}'.format(zeff_cmbkxunwiseblue))
print('CMB lensing x unwisegreen: {}'.format(zeff_cmbkxunwisegreen))
print('CMB lensing x skao1: {}'.format(get_zeff(skao1g, cmbk, cosmo, z_cmbk)))

print('DESI LRGs: {}'.format(zeff_desilrg))
print('CMB lensing x DESI LRGs: {}'.format(zeff_cmbkxdesilrg))




plt.figure(1, figsize=(1.618 * 3.75, 3.75))

for sname in samples.keys():

    if sname=='CMB Lensing':
        plt.plot(z_gals, get_kernel(samples[sname], cosmo, z_gals), '-.', label=sname)
    elif sname=='DES-Y3 Sources':
        plt.plot(z_gals, get_kernel(samples[sname], cosmo, z_gals), 'k-', label=sname)
    else:
        plt.plot(z_gals, get_kernel(samples[sname], cosmo, z_gals), label=sname, alpha=0.2)


plt.legend()
# plt.xscale('log')
# plt.show()
plt.xlabel('Redshift $z$')
plt.ylabel('Kernel $W(z)$')
plt.xlim([0., 3.])
plt.ylim([0., 1.1])
plt.savefig('./plots/kernels.png', dpi=300, bbox_inches='tight')

# print(zeff_cmbk, zeff_desg, zeff_skao1g, zeff_cmbkxdesg, zeff_skao1g)

# print('blue: {}, green: {}, desi: {}'.format(zeff_unwiseblue, zeff_unwisegreen, zeff_desilrg))#, zeff_cmbkxunwiseblue, zeff_cmbkxunwisegreen)
# print(zeff_cmbkxunwiseblue, zeff_cmbkxunwisegreen)
# z_arr = np.logspace(-4, np.log10(1100), 2048)
# chi_arr = cosmo.comoving_radial_distance(1. / (1. + z_arr))
# chistar = cosmo.comoving_radial_distance(1. / (1. + 1100))

# Hz = cosmo.h_over_h0(1. / (1. + z_arr)) * cosmo['h'] * 100.

# W_kappa = (oneover_chmpc)**2. * 1.5 * (cosmo['Omega_b'] + cosmo['Omega_c']) \
#             * (cosmo['h'])**2. * (1. + z_arr) \
#             * chi_arr * (chistar - chi_arr) / (chistar * Hz)

# kern = W_kappa * W_kappa / chi_arr**2
# zeff = np.trapz(kern * z_arr,x=chi_arr) / np.trapz(kern, x=chi_arr)

