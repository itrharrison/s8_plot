import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.patches import Rectangle
from palettable import colorbrewer, wesanderson

from matplotlib.lines import Line2D

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)
# rc('axes', prop_cycle=(cycler('color', wesanderson.Zissou_5.mpl_colors)))
# rc('axes', prop_cycle=(cycler('color', colorbrewer.diverging.Spectral_4.mpl_colors)))
ibm_rgb_tuples = [(100/255, 143/255, 255/255),
                  (120/255, 94/255, 240/255),
                  (220/255, 38/255, 127/255),
                  (254/255, 97/255, 0/255),
                  (255/255, 176/255, 0/255)]
rc('axes', prop_cycle=(cycler('color', ibm_rgb_tuples)))

rect_cmb = Rectangle((0.01, 20), 0.1-0.01, 1200-20, color='C0', alpha=0.4)
rect_cmbk = Rectangle((0.01, 0.5), 0.2-0.01, 5-0.5, color='C0', alpha=0.4)
rect_galg = Rectangle((0.05, 0.09), 10-0.05, 1-0.09, color='C0', alpha=0.4)

rect_gal = Rectangle((0.01, 0.2), 0.4-0.01, 2-0.2, color='C1', alpha=0.4)
rect_lya = Rectangle((0.1, 2.2), 10-0.1, 5-2.2, color='C1', alpha=0.4)

rect_cmbkgal = Rectangle((0.01, 0.3), 0.3-0.01, 3-0.3, color='C2', alpha=0.4)
rect_cmbkgalg = Rectangle((0.03, 0.25), 1-0.03, 0.6-0.25, color='C2', alpha=0.4)

plt.close('all') # tidy up any unshown plots

plt.figure(1, figsize=(4.5, 3.75))

plt.xscale('log')
plt.yscale('log')

plt.xlim([0.01, 10])
plt.ylim([0.09, 1200])

plt.gca().add_patch(rect_cmb)
plt.gca().add_patch(rect_cmbk)
plt.gca().add_patch(rect_galg)

plt.gca().add_patch(rect_gal)
plt.gca().add_patch(rect_lya)

# plt.gca().add_patch(rect_cmbkgal)
# plt.gca().add_patch(rect_cmbkgalg)

data = np.genfromtxt('./data/s8_zeff_keff.txt', delimiter=',', names=True, dtype=['S128', float, float, float, float, float, float, 'S4'])

plt.scatter(data['k_eff'], data['z_eff'], s=50*data['S_8']/data['S_8'].max())
plt.xlabel('$k\,$[Mpc]$^{-1}$')
plt.ylabel('$z_{\\rm eff}$')
plt.savefig('./plots/s8_pae.png', dpi=320, bbox_inches='tight')

planck = np.where(data['z_eff']==data['z_eff'].max())

cmblensing = data[data['type']==b'cl']
cosmicshear = data[data['type']==b'cs']
cmblcosmicshear = data[data['type']==b'clcs']
cmblgals = data[data['type']==b'clg']
lya = data[data['type']==b'lya']
gals = data[data['type']==b'gc']

plt.figure(2, figsize=(1.618 * 3.75, 3.75))

plt.errorbar(cmblensing['z_eff'], cmblensing['S_8'], yerr=[cmblensing['S_8_lower'], cmblensing['S_8_upper']], fmt=',', c='C0')
plt.scatter(cmblensing['z_eff'], cmblensing['S_8'], s=5*data['k_eff'].max()/cmblensing['k_eff'], c='C0')

plt.errorbar(cosmicshear['z_eff'], cosmicshear['S_8'], yerr=[cosmicshear['S_8_lower'], cosmicshear['S_8_upper']], fmt=',', c='C1')
plt.scatter(cosmicshear['z_eff'], cosmicshear['S_8'], s=5*data['k_eff'].max()/cosmicshear['k_eff'], c='C1')

plt.errorbar(cmblcosmicshear['z_eff'], cmblcosmicshear['S_8'], yerr=[cmblcosmicshear['S_8_lower'], cmblcosmicshear['S_8_upper']], fmt=',', c='C2', alpha=0.4)
plt.scatter(cmblcosmicshear['z_eff'], cmblcosmicshear['S_8'], s=5*data['k_eff'].max()/cmblcosmicshear['k_eff'], c='C2', alpha=0.4)

plt.errorbar(cmblcosmicshear['z_eff']+0.04, cmblcosmicshear['S_8'], yerr=0.027, fmt=',', c='C2',)
plt.scatter(cmblcosmicshear['z_eff']+0.04, cmblcosmicshear['S_8'], s=5*data['k_eff'].max()/cmblcosmicshear['k_eff'], c='C2', facecolors='none')

plt.errorbar(cmblgals['z_eff'], cmblgals['S_8'], yerr=[cmblgals['S_8_lower'], cmblgals['S_8_upper']], fmt=',', c='C3')
plt.scatter(cmblgals['z_eff'], cmblgals['S_8'], s=5*data['k_eff'].max()/cmblgals['k_eff'], c='C3')

plt.errorbar(lya['z_eff'], lya['S_8'], yerr=[lya['S_8_lower'], lya['S_8_upper']], fmt=',', c='C4')
plt.scatter(lya['z_eff'], lya['S_8'], s=5*data['k_eff'].max()/lya['k_eff'], c='C4')

plt.errorbar(gals['z_eff'], gals['S_8'], yerr=[gals['S_8_lower'], gals['S_8_upper']], fmt=',', c='C3', alpha=0.4)
plt.scatter(gals['z_eff'], gals['S_8'], s=5*data['k_eff'].max()/gals['k_eff'], c='C3', alpha=0.4)

# plt.errorbar(data['z_eff'], data['S_8'], yerr=[data['S_8_lower'], data['S_8_upper']], fmt=',')
# plt.scatter(data['z_eff'][data['type']==b'cs'], data['S_8'][data['type']==b'cs'], s=5*data['k_eff'].max()/data['k_eff'][data['type']==b'cs'], c='C1')
# plt.scatter(data['z_eff'][data['type']==b'cl'], data['S_8'][data['type']==b'cl'], s=5*data['k_eff'].max()/data['k_eff'][data['type']==b'cl'], c='C2')
# plt.scatter(data['z_eff'], data['S_8'], s=5*data['k_eff'].max()/data['k_eff'])
plt.xscale('log')
plt.xlabel('$z_{\\rm eff}$')
plt.ylabel('$S_8 = \sigma_8 \sqrt{\Omega_{\\rm m}/0.3}$')
plt.fill_between([0., 2000.], data['S_8'][planck] - data['S_8_lower'][planck], data['S_8'][planck] + data['S_8_upper'][planck], alpha=0.2, zorder=-1000, color='k')

# plt.errorbar(0.36, 0.80, yerr=[0.02], fmt=',', c='C1')
# plt.scatter(0.36, 0.80, s=5*data['k_eff'].max()/cosmicshear['k_eff'][0], c='C1', facecolors='none')

# plt.errorbar(0.65, 0.80, yerr=[0.02], fmt=',', c='C2')
# plt.scatter(0.65, 0.80, s=5*data['k_eff'].max()/cmblcosmicshear['k_eff'][0], c='C2', facecolors='none')

# plt.fill_between([0.62, 0.68], 0.7, 0.9, zorder=-1000, alpha=0.2, color='C2')
# plt.fill_between([0.35, 0.37], 0.7, 0.9, zorder=-1000, alpha=0.2, color='C1')

# plt.xlim([0., 1500.])
plt.xlim([0., 7.])

# plt.text(0.3, 0.715, 'SKAO-1 Cosmic Shear', size='x-small', color='C1', rotation='vertical')
# plt.text(0.45, 0.715, 'CMB Lensing x\nSKAO-1 Cosmic Shear', size='x-small', color='C2', rotation='vertical')

plt.text(6.6, 0.825, 'Primary CMB \n at $z=1100$', ha='right', color='k')
plt.text(1.5e-1, 0.715, 'Cosmic\nShear', color='C1', size='x-small')
plt.text(2.75e-1, 0.705, 'CMB Lensing $\\times$\nCosmic Shear', color='C2', size='x-small')
plt.text(4.e-1, 0.76, 'CMB Lensing $\\times$\nGalaxy Clustering', color='C3', size='x-small')
plt.text(1.3, 0.785, 'CMB\nLensing', color='C0', size='x-small')
plt.text(3.5, 0.74, 'Ly-$\\alpha$', color='C4', size='x-small')
plt.text(5.8e-1, 0.88, 'Galaxy\nClustering', color='C3', size='x-small', alpha=0.4, ha='right')
plt.ylim([0.7, 0.9])

global_kmin = data['k_eff'].min()
global_kmax = data['k_eff'].max()

global_kmin_size = 5*data['k_eff'].max()/global_kmin
global_kmax_size = 5*data['k_eff'].max()/global_kmax

legend_elements = [Line2D([0], [0], lw=0, marker='o', markerfacecolor='k', markeredgecolor='k', markersize=np.sqrt(250), label='$k_{\\rm eff} = '+'{:.2f}'.format(global_kmin)+'\,$Mpc$^{-1}$'),
                   Line2D([0], [0], lw=0, marker='o', markerfacecolor='k', markeredgecolor='k', markersize=np.sqrt(5), label='$k_{\\rm eff} = '+'{:.2f}'.format(global_kmax)+'\,$Mpc$^{-1}$')]

plt.legend(handles=legend_elements, fontsize='small')

plt.savefig('./plots/s8_zeff.png', dpi=320, bbox_inches='tight')
plt.savefig('./plots/s8_zeff.pdf', bbox_inches='tight')
