import os
from pyTSEB import TSEB
from pyTSEB import MO_similarity as mo
from pyTSEB import wind_profile as wind
from pyTSEB import resistances as res
from pyTSEB import energy_combination_ET as pet
from pyTSEB import meteo_utils as met
from pyTSEB import net_radiation as rad
from pyTSEB import clumping_index as ci
from pypro4sail import machine_learning_regression as inv
from pypro4sail import prospect as pro
from pypro4sail import four_sail as sail
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import ipywidgets as w
from IPython.display import display, clear_output
print("Gracias! librerías correctamente importadas")
print("Puedes continuar con las siguientes tareas")

slide_kwargs = {"continuous_update": False}
FIGSIZE = (12.0, 6.0)

np.seterr(all="ignore")
INPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
CIMIS_FILE_PATH = os.path.join(INPUT_FOLDER, "meteo",
                               "meteo_olive.csv")
SOIL_FOLDER = os.path.join(INPUT_FOLDER, "soil_spectral_library")
ALBEDO_WEIGHT_FILE = os.path.join(INPUT_FOLDER, "albedo_weights.csv")
CIMIS_DATA = pd.read_csv(CIMIS_FILE_PATH)

N_SIM = 50
N_SIM = 50
pet.ITERATIONS = 5
# Generate the list with VZAs (from 0 to 89)
FLUX_LIMS = 0, 500
ET_LIMS = 0, 10
LAI = 1.5
H_C = 0.12  # Canopy height
LEAF_WIDTH = 0.1  # Leaf width (m)
# Create list of heights
EMIS_C = 0.98
EMIS_S = 0.05
# Calculations are carried out for the following meteorological conditions: R, =
# 400 W m-2; D = 0,10,20 mb; T, = 25 “C; and u = 2ms-’. Such meteorological conditions
# might be considered typical for midday in the middle of a growing season at a subtropical
# site. However, the objective is not to make detailed predictions for particular meteoro-
# logical conditions, it is rather to illustrate the general features of the theoretical treatment
# described.
SDN = 300.  # We assume sn=400 to try to approach rn=400 when computing ln in the module
TAIR = 25 + 273.15  # K
PRESS = 1013.  # mb
TIME_START = 0
TIME_END = 24
LATITUDE = 40
HOURS = np.linspace(TIME_START, TIME_END, N_SIM)

ALBEDO_WEIGHTS = pd.read_csv(ALBEDO_WEIGHT_FILE)

LAIS = np.linspace(0, 6, N_SIM)
pet.ITERATIONS = 5
# Generate the list with VZAs (from 0 to 89)
VZAS = np.arange(0, 89)
DELTA_T_LIMS = -5, 20
FLUX_LIMS = 0, 500
ET_LIMS = 0, 10
U = 5.  # set wind speed (measured at 10m above the canopy)
Z_U = 10.
Z_T = 10.
LAI_REF = 24 * 0.12
H_C_REF = 0.12  # Canopy height
LEAF_WIDTH = 0.05  # Leaf width (m)
# Create list of heights
ZS = np.linspace(0, Z_U, N_SIM)
US = np.linspace(0.50, 20, N_SIM)
EMIS_C = 0.98
EMIS_S = 0.05
RES_FORM = [TSEB.KUSTAS_NORMAN_1999, {}]
# mean stomatal resistance, rsT, is taken as 400sm-1.  It follows from Eq. (19)
# that, for a leaf area index, L , of 4, the bulk stomatal resistance is 50 sm-1
RST_MIN = 400
GST_REF = 0.415
# Calculations are carried out for the following meteorological conditions: R, =
# 400 W m-2; D = 0,10,20 mb; T, = 25 “C; and u = 2ms-’. Such meteorological conditions
# might be considered typical for midday in the middle of a growing season at a subtropical
# site. However, the objective is not to make detailed predictions for particular meteoro-
# logical conditions, it is rather to illustrate the general features of the theoretical treatment
# described.
SDN = 300.  # We assume sn=400 to try to approach rn=400 when computing ln in the module
TAIR = 25 + 273.15  # K
PRESS = 1013.  # mb
VPD = 0.5 * met.calc_vapor_pressure(TAIR)  # mb
# For bare soil zb is commonly taken as 0.01 m (see Van Bavel and Hillel 1976)
Z0_SOIL = 0.01


CIMIS_DATA['Date'] = pd.to_datetime(CIMIS_DATA['TIMESTAMP_START'], format="%Y%m%d%H%M")
CIMIS_DATA["VPD_mean"] = 10 * CIMIS_DATA["VPD_F"]

w_kappa = w.FloatSlider(min=0, max=1, value=0.5,
                        description='$\kappa$', **slide_kwargs)
w_cab = w.FloatSlider(value=inv.MEAN_CAB,
                      min=inv.MIN_CAB,
                      max=inv.MAX_CAB,
                      step=1, description='Cab ($\mu$g/cm²)', **slide_kwargs)
w_lai = w.FloatSlider(value=inv.MEAN_LAI,
                      min=inv.MIN_LAI,
                      max=inv.MAX_LAI,
                      step=0.1, description='LAI (m²/m²)', **slide_kwargs)
w_leaf_angle = w.FloatSlider(value=57, min=1, max=90, step=1,
                             description='Leaf Angle (deg.)', **slide_kwargs)
w_sza = w.FloatSlider(value=37, min=0, max=89, step=1,
                      description='Sun Zenith Angle (deg.)', **slide_kwargs)
w_saa = w.FloatSlider(value=180, min=0, max=359, step=1,
                      description='Sun Azimuth Angle (deg.)', **slide_kwargs)
w_hc = w.FloatSlider(min=0.01, max=8, value=H_C, step=0.01, description='Altura dosel',
                      **slide_kwargs)
w_hb_ratio = w.FloatSlider(min=0.0, max=0.9, value=0.5, step=0.01,
                           description='Inicio del dosel',
                           **slide_kwargs)

w_wc = w.FloatSlider(min=0.1, max=3, value=1, step=0.01, description='Forma',
                     **slide_kwargs)
w_fc = w.FloatSlider(min=0.05, max=1, value=0.2, step=0.01,
                     description='Cobertura del suelo', **slide_kwargs)
w_interrow = w.FloatSlider(min=0.1, max=6, value=1, step=0.1,
                           description='Separación hileras', **slide_kwargs)
w_psi = w.FloatSlider(min=-90., max=90, value=0, description='Orientación hilera',
                      **slide_kwargs)
w_sdn = w.FloatSlider(min=0., max=400, value=300,
                      description='Irradiancia diaria (W/m²)', **slide_kwargs)
w_skyl = w.FloatSlider(min=0, max=1, value=0.1, step=0.01,
                       description='Fracción difusa',**slide_kwargs)
w_leaf_abs = w.FloatSlider(min=0.8, max=1, value=0.9, step=0.01,
                           description='Absortividad hoja', **slide_kwargs)

w_tair = w.FloatSlider(min=273, max=313, value=293, step=0.1,
                       description='T$_{aire}$ (K)', **slide_kwargs)
w_hr = w.FloatSlider(min=0, max=100, value=50, step=0.1,
                       description='HR (%)', **slide_kwargs)
w_emiss = w.FloatSlider(min=0.9, max=0.99, value=0.98, step=0.01,
                        description='$\epsilon$', **slide_kwargs)
w_lat = w.FloatSlider(min=0.0, max=90, value=40, step=1,
                        description='Latitud absoluta (deg.)', **slide_kwargs)

w_zol = w.FloatSlider(min=-1, max=1, value=0, step=0.01, description='Estabilidad',
                      **slide_kwargs)
w_u = w.FloatSlider(min=0.1, max=20, value=U, step=0.1, description='Velocidad del viento (m/s)',
                    **slide_kwargs)

w_r_ss = w.FloatSlider(min=0, max=10000, value=2000, step=100,
                       description='R$_{ss}$ (s/m)', **slide_kwargs)
w_g_st = w.FloatSlider(min=0, max=0.5, value=GST_REF, step=0.001,
                       description='g$_{st}$ (mmol/m²s¹)', **slide_kwargs)
w_lai_range = w.FloatRangeSlider(min=0, max=10, value=[0, 4], step=0.1,
                                 description='LAI', **slide_kwargs)
w_vza = w.FloatSlider(min=0, max=89, value=0, step=1,
                       description='VZA (deg.)', **slide_kwargs)
w_ev = w.FloatSlider(min=0.97, max=1, value=0.99, step=0.001,
                     description='$\epsilon_V$', readout_format='.3f',
                     **slide_kwargs)
w_es = w.FloatSlider(min=0.90, max=1, value=0.97, step=0.001,
                     description='$\epsilon_S$', readout_format='.3f',
                     **slide_kwargs)


def beer_lambert_law(kappa, length):
     tau = np.exp(-kappa * length)
     plot_attenuation(length, tau)

def plot_attenuation(length, tau):
    fig, axs = plt.subplots(figsize=FIGSIZE)
    # Create an empty list with the returning f_c for each canopy
    axs.plot(length, tau, 'k-')
    axs.set_xlabel('Path length')
    axs.set_ylabel('Trasmittance')
    axs.set_ylim((0, 1))
    axs.set_xlim((np.min(length), np.max(length)))
    plt.tight_layout()
    plt.show()


def plot_fipar(lai, leaf_angle):
    fig, axs = plt.subplots(figsize=FIGSIZE)
    # Create an empty list with the returning f_c for each canopy
    chi = rad.leafangle_2_chi(leaf_angle)
    szas = np.arange(0, 90)
    fipar = TSEB.calc_F_theta_campbell(szas, lai, x_LAD=chi)
    # Plot the results
    axs.plot(90 - szas, fipar, 'r')
    axs.set_xlabel('Solar Elevation Angle (degrees)')
    axs.set_ylabel('Fraction of Interecptad Radiation')
    axs.set_ylim((0, 1.05))
    axs.set_xlim((0, 90))
    plt.tight_layout()
    plt.show()


def solar_irradiance(sdn, szas):
    rdirvis, rdifvis, rdirnir, rdifnir = rad.calc_potential_irradiance_weiss(szas,
                                                                             press=np.full(N_SIM, 1013.15))
    sdn_pot = rdirvis + rdifvis + rdirnir + rdifnir
    sdn = sdn * np.size(sdn_pot) * sdn_pot / np.sum(sdn_pot)
    return sdn


def plot_net_solar_radiation(local_lai, leaf_angle, h_c, f_c, row_distance,
                             row_direction, sdn_day, skyl=0.1, fvis=0.55,
                             lat=LATITUDE):
    chi = rad.leafangle_2_chi(leaf_angle)
    lai = np.full(N_SIM, local_lai) * f_c
    szas, saas = met.calc_sun_angles(np.full(N_SIM, lat),
                                     np.zeros(N_SIM),
                                     np.zeros(N_SIM),
                                     np.full(N_SIM, 180),
                                     HOURS)
    sdn = solar_irradiance(sdn_day, szas)
    sdn_dir = (1. - skyl) * sdn
    sdn_dif = skyl * sdn
    w_c = row_distance * f_c / np.full(N_SIM, h_c)
    psi = relative_azimuth(row_direction, saas)
    omega = ci.calc_omega_rows(np.full(N_SIM, lai),
                               np.full(N_SIM, f_c),
                               theta=szas,
                               psi=psi,
                               w_c=w_c,
                               x_lad=np.full(N_SIM, chi))
    lai_eff = local_lai * omega
    sn_v, sn_s = rad.calc_Sn_Campbell(lai,
                                      szas,
                                      sdn_dir,
                                      sdn_dif,
                                      fvis,
                                      1. - fvis,
                                      np.full(N_SIM, 0.07),
                                      np.full(N_SIM, 0.07),
                                      np.full(N_SIM, 0.35),
                                      np.full(N_SIM, 0.35),
                                      np.full(N_SIM, 0.15),
                                      np.full(N_SIM, 0.25),
                                      x_LAD=chi,
                                      LAI_eff=lai_eff)

    sn = sn_v + sn_s
    albedo = 1. - sn / sdn
    fig, axs = plt.subplots(figsize=FIGSIZE)
    # Plot the results
    axs.plot(HOURS, sn, 'black', label="Net Radiation")
    axs.plot(HOURS, sn_v, 'green', label="Net Canopy Radiation")
    axs.plot(HOURS, sn_s, 'orange', label="Net Soil Radiation")
    axs.set_xlabel('Time')
    axs.set_ylabel('Solar Radiation (W/m²)')
    axs.set_xlim((TIME_START, TIME_END))
    axs.set_ylim((0, 1000))
    axs.legend()
    secax = axs.twinx()
    secax.plot(HOURS, albedo, "b")
    secax.set_ylabel('Albedo', color="blue")
    secax.tick_params(axis='y', colors='blue')
    secax.set_ylim((0, 1))
    plt.show()


def plot_apar(local_lai, leaf_angle, leaf_absorbance, h_c, f_c, row_distance,
              row_direction, sdn_day, soil_albedo=0.15, skyl=0.1, fvis=0.55,
              lat=LATITUDE):
    szas, saas = met.calc_sun_angles(np.full(N_SIM, lat),
                                     np.zeros(N_SIM),
                                     np.zeros(N_SIM),
                                     np.full(N_SIM, 180),
                                     HOURS)
    # Tweak for introducing the calc_spectra_Cambpell inputs
    rho_leaf_vis = 0.5 * (1 - np.full(N_SIM, leaf_absorbance))
    tau_leaf_vis = 0.5 * (1 - np.full(N_SIM, leaf_absorbance))
    chi = rad.leafangle_2_chi(leaf_angle)
    lai = np.full(N_SIM, local_lai) * f_c
    sdn = solar_irradiance(sdn_day, szas)
    sdn_dir = (1. - skyl) * sdn
    sdn_dif = skyl * sdn
    w_c = row_distance * f_c / np.full(N_SIM, h_c)
    omega = ci.calc_omega_rows(np.full(N_SIM, lai),
                               np.full(N_SIM, f_c),
                               theta=szas,
                               psi=row_direction - saas,
                               w_c=w_c,
                               x_lad=np.full(N_SIM, chi))
    lai_eff = local_lai * omega
    albb, albd, taubt, taudt = rad.calc_spectra_Cambpell(lai,
                                                         szas,
                                                         rho_leaf_vis,
                                                         tau_leaf_vis,
                                                         np.full(N_SIM, soil_albedo),
                                                         x_lad=chi,
                                                         lai_eff=lai_eff)
    apar = (1.0 - taubt) * (1.0 - albb) * sdn_dir * fvis + \
           (1.0 - taudt) * (1.0 - albd) * sdn_dif * fvis

    akb = rad.calc_K_be_Campbell(szas, chi, radians=False)
    taub = np.exp(-akb * lai_eff)
    taud = rad._calc_taud(chi, lai)
    ipar = (1.0 - taub) * sdn_dir * fvis + (1.0 - taud) * sdn_dif * fvis
    par = sdn_dir * fvis + sdn_dif * fvis
    # Plot the results
    fig, axs = plt.subplots(figsize=FIGSIZE)
    axs.plot(HOURS, par, 'black', label="Entrante")
    axs.plot(HOURS, apar, 'green', label="Absorbida")
    axs.plot(HOURS, ipar, 'red', label="Interceptada")
    axs.set_xlabel('Time')
    axs.set_ylabel('PAR (W/m²)')
    axs.set_xlim((TIME_START, TIME_END))
    axs.set_ylim((0, 1000))
    axs.legend()
    plt.show()


def relative_azimuth(angle1, angle2, degrees=True):
    if degrees:
        ref = 180.
        divisor = 360.
    else:
        ref = np.pi
        divisor = 2 * np.pi
    psi = ref - np.abs(ref - np.mod(np.abs(angle1 - angle2), divisor))
    return psi


def bidirectional_reflectance(cab, lai, leaf_angle, sza=35., saa=180., skyl=0.1):
    wls_sim = ["PAR", "SW"]
    vzas = np.linspace(0, 90, num=50)
    vaas = np.linspace(0, 360, num=60)
    step_vza = np.radians(vzas[1] - vzas[0])
    step_psi = np.radians(vaas[1] - vaas[0])
    vzas, vaas = np.meshgrid(vzas, vaas, indexing="ij")
    n_sims = np.size(vzas)
    wls, r, t = pro.prospectd(inv.MEAN_N_LEAF, cab, inv.MEAN_CAR,
                              inv.MEAN_CBROWN, inv.MEAN_CW, inv.MEAN_CM,
                              inv.MEAN_ANT)
    rho_leaf = []
    tau_leaf = []
    rho_soil = []
    cosvza = np.cos(np.radians(vzas.reshape(-1)))
    sinvza = np.sin(np.radians(vzas.reshape(-1)))
    rsoil = np.genfromtxt(os.path.join(SOIL_FOLDER, "alfisol.fragiboralf.txt"))
    rho_leaf.append(np.sum(r * ALBEDO_WEIGHTS["w_PAR"]))
    tau_leaf.append(np.sum(t * ALBEDO_WEIGHTS["w_PAR"]))
    rho_soil.append(np.sum(rsoil[:, 1] * ALBEDO_WEIGHTS["w_PAR"]))
    # rho_leaf.append(np.sum(r * ALBEDO_WEIGHTS["NIR"]))
    # tau_leaf.append(np.sum(t * ALBEDO_WEIGHTS["NIR"]))
    # rho_soil.append(np.sum(rsoil[:, 1] * ALBEDO_WEIGHTS["NIR"]))
    rho_leaf.append(np.sum(r * ALBEDO_WEIGHTS["w_SW"]))
    tau_leaf.append(np.sum(t * ALBEDO_WEIGHTS["w_SW"]))
    rho_soil.append(np.sum(rsoil[:, 1] * ALBEDO_WEIGHTS["w_SW"]))
    lidf = sail.calc_lidf_campbell_vec(np.full(n_sims, leaf_angle))


    rho_leaf = np.tile(rho_leaf, (n_sims, 1))
    tau_leaf = np.tile(tau_leaf, (n_sims, 1))
    rho_soil = np.tile(rho_soil, (n_sims, 1))
    psi = relative_azimuth(vaas.reshape(-1), saa)

    [_,
     _,
     _,
     _,
     _,
     _,
     _,
     _,
     _,
     _,
     _,
     _,
     _,
     _,
     rdot,
     _,
     _,
     rsot,
     _,
     _,
     _] = sail.foursail_vec(np.full(n_sims, lai),
                            np.full(n_sims, inv.MEAN_HOTSPOT),
                            lidf,
                            np.full(n_sims, sza),
                            vzas.reshape(-1),
                            psi,
                            rho_leaf.T,
                            tau_leaf.T,
                            rho_soil.T)


    r2 = rdot * skyl + rsot * (1 - skyl)
    if lai == 0:
        r2 = rho_soil.T
    albedo = np.sum(r2 * cosvza * sinvza * step_vza * step_psi / np.pi, axis=1)
    polar_plot(vzas, vaas, r2, sza, saa, wls_sim, albedo)

    return wls_sim, r2

def polar_plot(vzas, vaas, rhos, sza, saa, wls_sim, albedo):
    fig, ax = plt.subplots(ncols=len(wls_sim), figsize=FIGSIZE,
                           subplot_kw={'projection': 'polar'})
    for i, wl in enumerate(wls_sim):
        ax[i].set_theta_direction(-1)
        ax[i].set_theta_offset(np.pi / 2.0)
        im = ax[i].pcolormesh(np.radians(vaas), vzas, rhos[i].reshape(vzas.shape),
                               vmin=0, vmax=0.25, cmap="inferno", shading='auto')
        ax[i].plot(np.radians(saa), sza, marker="*", markersize=12,
                   markerfacecolor="none", markeredgecolor="black")
        ax[i].grid(True)
        ax[i].set_title(f"{wls_sim[i]}\nalbedo={albedo[i]:.2f}")

    plt.tight_layout()
    cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.03])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    plt.show()


def plot_longwave_radiation(t_air, hr, delta_t, emiss=0.98):
    ea = hr * met.calc_vapor_pressure(t_air)
    emiss_atm = rad.calc_emiss_atm(ea, t_air)
    l_sky = emiss_atm * met.calc_stephan_boltzmann(t_air)
    lst = t_air + delta_t
    ln = emiss * (l_sky - met.calc_stephan_boltzmann(lst))
    plt.figure(figsize=FIGSIZE)
    plt.plot(lst, ln, "k-")
    plt.xlabel("LST (K)")
    plt.xlim((np.min(lst), np.max(lst)))
    plt.ylabel("Net longwave radiation (W/m²)")
    plt.ylim((0, 400))
    plt.tight_layout()
    plt.show()
    return ln


def plot_fveg(lai, leaf_angle=57):
    chi = rad.leafangle_2_chi(leaf_angle)
    # Create an empty list with the returning f_c for each canopy
    fc = TSEB.calc_F_theta_campbell(VZAS, lai, x_LAD=chi)
    fc_sph = TSEB.calc_F_theta_campbell(VZAS, lai, x_LAD=1)
    # Plot the results
    fig, axs = plt.subplots(figsize=FIGSIZE)
    axs.plot(VZAS, fc, 'r')
    axs.plot(VZAS, fc_sph, 'k--', label='Spherical')
    axs.set_xlabel('VZA (degrees)')
    axs.set_ylabel('Crop Fraction Observed by the Sensor')
    axs.set_ylim((0, 1.05))
    axs.set_xlim((0, 90))
    axs.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    return fc


def calc_roughness(h_c):
    z_0m = h_c / 8.
    d_0 = 2. * h_c / 3.
    return z_0m, d_0


def l_2_zol(z_0m, zol):
    if zol == 0.0:
        l_mo = np.inf
    else:
        l_mo = z_0m / zol
    return l_mo


def wind_profile(zol, lai, h_c=0.12):
    # Estimate surface roughness
    z_0m, d_0 = calc_roughness(h_c)
    l_mo = l_2_zol(z_0m, zol)
    # Calculate the friction velocity
    u_friction = mo.calc_u_star(U, Z_U, l_mo, d_0, z_0m)
    upper = ZS >= h_c
    u_z = np.ones(ZS.shape)
    u_c = wind.calc_u_C_star(u_friction, h_c, d_0, z_0m, l_mo)
    u_z[upper] = wind.calc_u_C_star(u_friction, ZS[upper], d_0, z_0m, l_mo)
    u_z[~upper] = wind.calc_u_Goudriaan(u_c, h_c, lai, LEAF_WIDTH, ZS[~upper])
    return u_z, u_c


def wind_profile_homogeneous(zol, lai, hc):
    u_z, u_c = wind_profile(zol, lai, h_c=hc)
    u_z_ref, _ = wind_profile(0., LAI_REF, h_c=H_C_REF)
    plot_profile(u_z, u_c, hc, u_z_ref=u_z_ref)
    return u_z, u_c


def wind_profile_heterogeneous(zol, lai, hc, hb_ratio=0.5):
    # Estimate surface roughness
    z_0m, d_0 = calc_roughness(hc)
    l_mo = l_2_zol(z_0m, zol)
    # Calculate the friction velocity
    u_friction = mo.calc_u_star(U, Z_U, l_mo, d_0, z_0m)
    upper = ZS >= hc
    u_z = np.ones(ZS.shape)
    u_c = wind.calc_u_C_star(u_friction, hc, d_0, z_0m, l_mo)
    u_z[upper] = wind.calc_u_C_star(u_friction, ZS[upper], d_0, z_0m, l_mo)
    h_b = hb_ratio * hc
    Xi_max, sigma_u, sigma_l = wind.canopy_shape(hc,
                                                 h_b=h_b,
                                                 h_max=0.5)
    f_a = wind.calc_canopy_distribution(Xi_max, sigma_u, sigma_l)
    f_a_cum = wind.calc_cummulative_canopy_distribution(f_a)
    u_z[~upper] = wind.calc_u_Massman(np.full(np.sum(~upper), u_c),
                                      np.full(np.sum(~upper), hc),
                                      np.full(np.sum(~upper), lai),
                                      ZS[~upper],
                                      f_a_cum,
                                      xi_soil=Z0_SOIL/hc)
    plot_profile(u_z, u_c, hc)
    plt.figure(figsize=FIGSIZE)
    plt.plot(f_a, np.linspace(0, hc, np.size(f_a)))
    plt.ylim((0, hc))
    plt.xlim((0, None))
    plt.xlabel('Foliar density')
    plt.ylabel('Height above ground (m)')
    plt.tight_layout()
    plt.show()
    return u_z, u_c


def plot_profile(u_z, u_c, hc, u_z_ref=None):
    # Plots the wind profile for given stability lenght compared to a neutral atmosphere
    fig, axs = plt.subplots(figsize=FIGSIZE)
    axs.plot(u_z, ZS, 'b', label="Wind profile")
    # Plot the ufriction wind canopy
    axs.plot(u_c, hc, marker='*', markerfacecolor="none", markeredgecolor ="blue",
             ms=12, ls="None", label='$u_c$')
    if u_z_ref is not None:
        axs.plot(u_z_ref, ZS, 'k--', label="FAO56 reference profile")
    # Plot the canopy windspeed according to the two different methods
    axs.legend(loc='upper left')
    axs.set_xlim((0, U))
    axs.set_ylim((0, Z_U))
    axs.set_xlabel('Wind Speed (m)')
    axs.set_ylabel('Height above ground (m)')
    plt.tight_layout()
    plt.show()


def plot_aerodynamic_resistance(zol, hc):
    def calc_r_a(zol, h_c):
        z_0m, d_0 = calc_roughness(np.full_like(ZS, h_c))
        if zol != 0:
            l_mo = z_0m / zol
        else:
            l_mo = np.inf
        u_friction = mo.calc_u_star(US,
                                    np.full_like(US, Z_U),
                                    l_mo,
                                    d_0,
                                    z_0m)
        ra = res.calc_R_A(Z_U, u_friction, l_mo, d_0, z_0m)
        return ra

    # Plots the resistances for a range of windspeeds
    fig, axs = plt.subplots(figsize=FIGSIZE)
    ra = calc_r_a(zol, hc)
    axs.plot(US, ra, 'k')
    axs.set_ylabel('Aerodynamic Resistance (s/m)')
    axs.set_ylim((0, 200))
    axs.set_xlabel('Wind speed (m/s)')
    axs.set_xlim((0, None))
    plt.tight_layout()
    plt.show()


def plot_resistances(lai, hc, l_mo, leaf_width, z0_soil, delta_t):
    def calc_resistances(lai, hc, l_mo, leaf_width, z0_soil, delta_t,
                        resistance_flag=0):
        rs_list = []
        rx_list = []
        ra_list = []
        z_0m = np.full_like(ZS, hc) / 8.
        d_0 = 2. * np.full_like(ZS, hc) / 2.
        for u in US:
            u_friction = mo.calc_u_star(u, Z_U, l_mo, d_0, z_0m)
            u_c = wind.calc_u_C_star(u_friction, hc, d_0, z_0m, l_mo)
            # Resistances
            if resistance_flag == 0:
                u_S = wind.calc_u_Goudriaan(u_c, hc, lai, leaf_width, z0_soil)
                u_d_zm = wind.calc_u_Goudriaan(u_c, hc, lai, leaf_width, d_0 + z_0m)
                rx = res.calc_R_x_Norman(lai, leaf_width, u_d_zm)
                rs = res.calc_R_S_Kustas(u_S, delta_t)
            elif resistance_flag == 1:
                rx = res.calc_R_x_Choudhury(u_c, lai, leaf_width)
                rs = res.calc_R_S_Choudhury(u_friction, hc, z_0m, d_0, Z_U,
                                             z0_soil)
            elif resistance_flag == 2:
                rx = res.calc_R_x_McNaughton(lai, leaf_width, u_friction)
                rs = res.calc_R_S_McNaughton(u_friction)
            elif resistance_flag == 3:
                alpha_k = wind.calc_A_Goudriaan(hc, lai, leaf_width)
                alpha_prime = float(alpha_k)
                rx = res.calc_R_x_Choudhury(u_c, lai, leaf_width,
                                            alpha_prime=alpha_prime)
                rs = res.calc_R_S_Choudhury(u_friction, hc, z_0m, d_0, Z_U,
                                            z0_soil, alpha_k=alpha_k)

            ra = res.calc_R_A(Z_U, u_friction, l_mo, d_0, z_0m)
            # Add the results to the ouput list
            rs_list.append(rs)
            rx_list.append(rx)
            ra_list.append(ra)

        return ra_list, rx_list, rs_list

    # Plots the resistances for a range of windspeeds
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=FIG_SIZE)
    ra, rx, rs = calc_resistances(lai, hc, l_mo, leaf_width, z0_soil,
                                  delta_t, resistance_flag=0)
    axs[0].plot(US, ra, 'k', label='KN99')
    axs[1].plot(US, rx, 'k', label='KN99')
    axs[2].plot(US, rs, 'k', label='KN99')
    ra, rx, rs = calc_resistances(lai, hc, l_mo, leaf_width, z0_soil,
                                  delta_t, resistance_flag=1)
    axs[0].plot(US, ra, 'r', label='CM88')
    axs[1].plot(US, rx, 'r', label='CM88')
    axs[2].plot(US, rx, 'r', label='CM88')
    ra, rx, rs = calc_resistances(lai, hc, l_mo, leaf_width, z0_soil,
                                  delta_t, resistance_flag=2)
    axs[0].plot(US, ra, 'b', label='MH95')
    axs[1].plot(US, rx, 'b', label='MH95')
    axs[2].plot(US, rx, 'b', label='MH95')
    ra, rx, rs = calc_resistances(lai, hc, l_mo, leaf_width, z0_soil,
                                  delta_t, resistance_flag=3)
    axs[0].plot(US, ra, 'g', label='N14')
    axs[1].plot(US, rx, 'g', label='N14')
    axs[2].plot(US, rs, 'g', label='N14')
    axs[0].legend(bbox_to_anchor=(0, 1), loc=3, ncol=4)
    axs[0].set_ylabel('Aerodynamie Resistance')
    axs[0].tick_params(axis='x', which='both', bottom='off', top='off',
                       labelbottom='off')
    axs[1].set_ylabel('Canopy Resistance')
    axs[1].tick_params(axis='x', which='both', bottom='off', top='off',
                       labelbottom='off')
    axs[2].set_ylabel('Soil Resistance')
    axs[2].set_xlabel('Wind speed')
    axs[0].set_ylim((0, 200))
    axs[1].set_ylim((0, 200))
    axs[2].set_ylim((0, 200))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()


def plot_flux_variation(values, le, le_c, le_pm, le_fao, t_c, t_s, t_0,
                        var="LAI"):
    fig, axs = plt.subplots(3, figsize=FIGSIZE, sharex=True)
    axs[0].plot(values, le, linestyle="-", color="blue", label="ET$_{SW}$")
    axs[0].plot(values, le_pm, linestyle="-", color="red", label="ET$_{PM}$")
    axs[0].plot(0.5 * LAI_REF, le_fao, color="black", markersize=12, marker="*", ls="none",
                label="ET$_{FAO56}$")

    axs[1].plot(values, le_c / le, linestyle="-", color="green", label="$\Delta$ET$_{C}$")
    axs[1].plot(values, 1 - le_c / le, linestyle="-", color="orange", label="\Delta$ET$_{S}$")


    axs[2].plot(values, t_0, linestyle="-", color="black", label="T$_0$ - T$_a$")
    axs[2].plot(values, t_c, linestyle="-", color="green", label="T$_c$ - T$_a$")
    axs[2].plot(values, t_s, linestyle="-", color="orange", label="T$_s$ - T$_a$")
    axs[2].axhline(0, c="silver", ls=":")

    value_lims = np.min(values), np.max(values)

    axs[0].legend()
    axs[2].legend()
    axs[0].set_ylabel("ET (mm / day)$)")
    axs[0].set_ylim(ET_LIMS)
    axs[1].set_ylabel("LAYER FRACTION")
    axs[1].set_ylim((0, 1))
    axs[2].set_ylabel("T$_x$ - T$_a$ (K)")
    axs[2].set_ylim(DELTA_T_LIMS)
    axs[0].set_xlim(value_lims)
    axs[1].set_xlim(value_lims)
    axs[2].set_xlim(value_lims)
    axs[2].set_xlabel(var)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()


def fluxes_and_resistances(g_st=GST_REF, r_ss=2000, h_c=H_C_REF):
    tair = np.full(N_SIM, TAIR)

    r_st = 1. / (TSEB.res.molm2s1_2_ms1(tair, PRESS) * g_st)
    sn = np.full(N_SIM, SDN) * (1. - 0.23)
    sn_s = sn * np.exp(-0.5 * LAIS)
    sn_c = sn - sn_s
    es = met.calc_vapor_pressure(tair)
    ea = es - VPD
    ldn = rad.calc_emiss_atm(ea, tair) * met.calc_stephan_boltzmann(tair)
    z_0m, d_0 = calc_roughness(np.full(N_SIM, h_c))
    [_, t_s, t_c, _, _, _, le, _, le_c, *_] = pet.shuttleworth_wallace(
        tair,
        U,
        ea,
        PRESS,
        sn_c,
        sn_s,
        ldn,
        LAIS,
        h_c,
        EMIS_C,
        EMIS_S,
        z_0m,
        d_0,
        Z_U,
        Z_T,
        leaf_width=LEAF_WIDTH,
        z0_soil=Z0_SOIL,
        Rst_min=r_st,
        R_ss=r_ss,
        resistance_form=RES_FORM,
        calcG_params=[[0], np.zeros(N_SIM)],
        leaf_type=1,
        verbose=False)

    le_c[0] = 0
    t_c[0] = np.nan

    [_, t_0, _, le_pm, *_] = pet.penman_monteith(tair,
                                                 U,
                                                 ea,
                                                 PRESS,
                                                 sn,
                                                 ldn,
                                                 EMIS_C,
                                                 LAIS,
                                                 z_0m,
                                                 d_0,
                                                 Z_U,
                                                 Z_T,
                                                 Rst_min=r_st,
                                                 calcG_params=[[0],
                                                               np.zeros(N_SIM)],
                                                 leaf_type=1,
                                                 verbose=False)

    le_fao = pet.pet_fao56(TAIR,
                           U,
                           ea[0],
                           es[0],
                           PRESS,
                           np.asarray(SDN),
                           Z_U,
                           Z_T,
                           f_cd=1,
                           is_daily=True)

    le_pm = met.flux_2_evaporation(le_pm, t_k=TAIR, time_domain=24)
    le_fao = met.flux_2_evaporation(le_fao, t_k=TAIR, time_domain=24)
    le = met.flux_2_evaporation(le, t_k=TAIR, time_domain=24)
    le_c = met.flux_2_evaporation(le_c, t_k=TAIR, time_domain=24)

    plot_flux_variation(LAIS, le, le_c, le_pm, le_fao,
                        t_c - TAIR, t_s - TAIR, t_0 - TAIR,
                        var="LAI")

    return t_c, t_s, t_0


def get_land_surface_temperature(vza, leaf_angle, temperatures, e_v=0.98, e_s=0.95):
    t_c, t_s, t_0 = temperatures.result
    bt_obs, emiss = lst_from_4sail(e_v, e_s, t_c, t_s, LAIS, vza, leaf_angle, t_atm=243.)
    chi = rad.leafangle_2_chi(leaf_angle)
    fc = TSEB.calc_F_theta_campbell(vza, LAIS, x_LAD=chi)
    lst = (fc * t_c**4 + (1. - fc) * t_s**4)**0.25
    bt_obs[LAIS == 0] = t_s[LAIS == 0]
    lst[LAIS == 0] = t_s[LAIS == 0]
    fig, axs = plt.subplots(nrows=2, figsize=FIGSIZE, sharex=True)
    axs[0].plot(LAIS, fc, 'k-')
    axs[0].set_ylabel('Crop Fraction Observed by the Sensor')
    axs[0].set_ylim((0, 1))
    axs[1].plot(LAIS, t_0, 'k-', label='T$_0$')
    axs[1].plot(LAIS, lst, 'r-', label='LST simplified')
    axs[1].plot(LAIS, bt_obs, 'r:', label='LST analytical')
    axs[1].set_xlabel('LAI')
    axs[1].set_ylabel('LST')
    axs[1].legend(loc='upper right')
    axs[1].set_ylim((TAIR - 5, TAIR + 20))
    axs[1].set_xlim((0, np.max(LAIS)))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()


def plot_kcs(dates, lais, et_ref, et, kcs):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=FIGSIZE, sharex="col")
    gs = axs[0, 1].get_gridspec()
    for ax in axs[:, 1]:
        ax.remove()
    for ax in axs[:, 2]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, 1:])
    axs[0, 0].plot(dates, lais, linestyle="-", color="green", lw=0.5)
    axs[1, 0].plot(dates, et_ref, linestyle="-", color="black", lw=0.5,
                   label="ET$_{ref}$")
    axs[1, 0].plot(dates, et, linestyle="-", color="blue", lw=0.5,
                   label="ET$_{a}$")
    axbig.scatter(lais, kcs, c="black", s=3, label="$Kc_{SW}$")

    axs[1, 0].legend()
    axs[0, 0].set_ylabel("LAI")
    axs[0, 0].set_ylim((0, 4))
    axs[1, 0].set_ylabel("ET (mm/day)")
    axs[1, 0].set_ylim(ET_LIMS)

    axs[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axs[1, 0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13, 3)))

    axbig.set_ylabel("Crop coefficient")
    axbig.set_ylim((0., 2))
    axbig.set_xlabel("LAI")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()


def build_day(doys, lai_range):
    doy_angle = np.pi * (doys - 180.) / 365
    lais = lai_range[0] + (lai_range[1] - lai_range[0]) * np.cos(doy_angle) ** 4
    return np.maximum(0, lais)


def crop_coefficients(g_st=GST_REF, r_ss=2000, h_c=H_C_REF, lai_range=(0, 5)):
    lais = build_day(CIMIS_DATA["Jul"], lai_range)
    sn = CIMIS_DATA["Sol Rad (W/sq.m)"].values * (1. - 0.23)
    sn_s = sn * np.exp(-0.5 * lais)
    sn_c = sn - sn_s
    tair = CIMIS_DATA["Avg Air Temp (C)"].values + 273.15
    r_st = 1. / (TSEB.res.molm2s1_2_ms1(tair, PRESS) * g_st)
    ea = 10 * CIMIS_DATA["Avg Vap Pres (kPa)"].values
    ldn = rad.calc_emiss_atm(ea, tair) * met.calc_stephan_boltzmann(tair)
    z_0m, d_0 = calc_roughness(np.full_like(sn, h_c))

    [_, t_s, t_c, _, _, _, le, _, le_c, *_] = pet.shuttleworth_wallace(
        tair,
        CIMIS_DATA["Avg Wind Speed (m/s)"].values,
        ea,
        PRESS,
        sn_c,
        sn_s,
        ldn,
        lais,
        h_c,
        EMIS_C,
        EMIS_S,
        z_0m,
        d_0,
        Z_U,
        Z_T,
        leaf_width=LEAF_WIDTH,
        z0_soil=Z0_SOIL,
        Rst_min=r_st,
        R_ss=r_ss,
        resistance_form=RES_FORM,
        calcG_params=[[0], np.zeros(sn.shape)],
        leaf_type=1,
        verbose=False)

    et = met.flux_2_evaporation(le, t_k=TAIR, time_domain=24)
    kcs_sw = et / CIMIS_DATA["ETo (mm)"].values
    out_file = os.path.join(OUTPUT_FOLDER, "lai_vs_kc.csv")
    if not os.path.isdir(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    result = pd.DataFrame({"LAI": lais, "Kc": kcs_sw})
    result.to_csv(out_file, index=False)
    plot_kcs(CIMIS_DATA["Date"], lais, CIMIS_DATA["ETo (mm)"],
             et, kcs_sw)


def lst_from_4sail(e_v, e_s, t_c, t_s, lais, vza, leaf_angle, t_atm=0):

    # Apply Kirchoff's law to get the soil and leaf bihemispherical reflectances
    rsoil = np.full((1, N_SIM), 1 - e_s)
    rho_leaf = np.full((1, N_SIM), 1 - e_v)
    tau_leaf = np.zeros((1, N_SIM))
    # Calculate the lidf,
    lidf = sail.calc_lidf_campbell_vec(np.full(N_SIM, leaf_angle))

    # 4SAIL for canopy reflectance and transmittance factors
    [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo, rso, rsos, rsod, rddt,
     rsdt, rdot,
     rsodt, rsost, rsot, gammasdf, gammasdb,
     gammaso] = sail.foursail_vec(lais,
                                np.full(N_SIM, 0.01),
                                lidf,
                                np.full(N_SIM, 37),
                                np.full(N_SIM, vza),
                                np.zeros(N_SIM),
                                rho_leaf, tau_leaf, rsoil)

    gammad = 1 - rdd - tdd
    gammao = 1 - rdo - tdo - too
    ttot = (too + tdo) / (1. - rsoil * rdd)
    gammaot = gammao + ttot * rsoil * gammad

    aeev = gammaot
    aees = ttot * e_v

    # Get the different canopy broadband emssion components
    h_vc = met.calc_stephan_boltzmann(t_c)
    h_gc = met.calc_stephan_boltzmann(t_s)
    h_sky = met.calc_stephan_boltzmann(t_atm)

    # Calculate the blackbody emission temperature
    lw = (rdot * h_sky + (aeev * h_vc + aees * h_gc )) / np.pi
    lst_obs = (np.pi * lw / rad.SB) ** (0.25)

    # Estimate the apparent surface directional emissivity
    emiss = 1 - rdot
    return lst_obs.reshape(-1), emiss.reshape(-1)

def rc_to_gst(rc, lai=0.5 * LAI_REF):
    rst = rc * lai
    gst = rst_to_gst(rst)
    return gst

def rst_to_gst(rst, t_c=293.15, p=1013.25):
    gst = 1. / (rst * res.molm2s1_2_ms1(t_c, p=p))
    return gst


