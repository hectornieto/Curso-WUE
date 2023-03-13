import os
from pyTSEB import TSEB
from pyTSEB import MO_similarity as mo
from pyTSEB import wind_profile as wind
from pyTSEB import resistances as res
from pyTSEB import energy_combination_ET as pet
from pyTSEB import meteo_utils as met
from pyTSEB import net_radiation as rad
from pypro4sail import four_sail as fs
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

# Generate the list with VZAs (from 0 to 89)
INPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
METEO_DAILY_FILE_PATH = os.path.join(INPUT_FOLDER, "meteo",
                                     "meteo_daily_olive.csv")

LAT_CND = 37.914998
LON_CND = -3.227659
STDLON_CND = 15.
ELEV_CND = 366.0
E_SURF = 0.98

N_SIM = 50
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

METEO_DAILY_DATA = pd.read_csv(METEO_DAILY_FILE_PATH, na_values=-9999)
METEO_DAILY_DATA['DATE'] = pd.to_datetime(METEO_DAILY_DATA['TIMESTAMP'],
                                          format="%Y%m%d").dt.date
METEO_DAILY_DATA['DOY'] = pd.to_datetime(METEO_DAILY_DATA['TIMESTAMP'],
                                         format="%Y%m%d").dt.day_of_year
METEO_DAILY_DATA = METEO_DAILY_DATA[
    ["DATE", "DOY", "TA_F", "VPD_F", "SW_IN_F", "LW_IN_F", "SW_OUT",
     "LW_OUT", "NETRAD", "WS_F", "PA_F", "LE_F_MDS", "H_F_MDS",
     "G_F_MDS"]]

METEO_DAILY_DATA["TA_F"] = METEO_DAILY_DATA["TA_F"] + 273.15
# Convert pressure units to mb
METEO_DAILY_DATA["ES"] = met.calc_vapor_pressure(METEO_DAILY_DATA["TA_F"].values)
METEO_DAILY_DATA["EA"] = METEO_DAILY_DATA["ES"].values - METEO_DAILY_DATA["VPD_F"].values
METEO_DAILY_DATA["PA_F"] = 10 * METEO_DAILY_DATA["PA_F"].values
METEO_DAILY_DATA["LE"] = METEO_DAILY_DATA['NETRAD'] - METEO_DAILY_DATA['G_F_MDS'] \
                         - METEO_DAILY_DATA['H_F_MDS']
METEO_DAILY_DATA["ET"] = met.flux_2_evaporation(METEO_DAILY_DATA["LE"],
                                                METEO_DAILY_DATA["TA_F"],
                                                24)
f_cd = pet.calc_cloudiness(METEO_DAILY_DATA["SW_IN_F"], LAT_CND, ELEV_CND, METEO_DAILY_DATA["DOY"])

METEO_DAILY_DATA["LE_ref"] = pet.pet_fao56(METEO_DAILY_DATA["TA_F"],
                                           METEO_DAILY_DATA["WS_F"],
                                           METEO_DAILY_DATA["EA"],
                                           METEO_DAILY_DATA["ES"],
                                           METEO_DAILY_DATA["PA_F"],
                                           METEO_DAILY_DATA["SW_IN_F"],
                                           Z_T,
                                           Z_U,
                                           f_cd=f_cd,
                                           is_daily=True)

METEO_DAILY_DATA["ET_ref"] = met.flux_2_evaporation(METEO_DAILY_DATA["LE_ref"],
                                                    METEO_DAILY_DATA["TA_F"],
                                                    24)
w_lai = w.FloatSlider(value=LAI_REF, min=0, max=10, step=0.1, description='LAI (m²/m²)',
                      **slide_kwargs)

w_leaf_angle = w.FloatSlider(value=57, min=1, max=90, step=1,
                             description='Leaf Angle (deg.)', **slide_kwargs)

w_zol = w.FloatSlider(min=-1, max=1, value=0, step=0.01, description='Estabilidad',
                      **slide_kwargs)
w_u = w.FloatSlider(min=0.1, max=20, value=U, step=0.1, description='Velocidad del viento (m/s)',
                    **slide_kwargs)

w_hc = w.FloatSlider(min=0.01, max=8, value=H_C_REF, step=0.01, description='Altura dosel',
                      **slide_kwargs)

w_hb_ratio = w.FloatSlider(min=0.0, max=0.9, value=0.5, step=0.01,
                           description='Inicio del dosel', **slide_kwargs)
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
    lais = build_day(METEO_DAILY_DATA["DOY"], lai_range)
    sn = METEO_DAILY_DATA["SW_IN_F"].values * (1. - 0.23)
    sn_s = sn * np.exp(-0.5 * lais)
    sn_c = sn - sn_s
    r_st = 1. / (TSEB.res.molm2s1_2_ms1(METEO_DAILY_DATA["TA_F"].values, METEO_DAILY_DATA["PA_F"].values) * g_st)
    ldn = rad.calc_emiss_atm(METEO_DAILY_DATA["EA"].values, METEO_DAILY_DATA["TA_F"].values) * met.calc_stephan_boltzmann(METEO_DAILY_DATA["TA_F"].values)
    z_0m, d_0 = calc_roughness(np.full_like(sn, h_c))

    [_, t_s, t_c, _, _, _, le, _, le_c, *_] = pet.shuttleworth_wallace(
        METEO_DAILY_DATA["TA_F"].values,
        METEO_DAILY_DATA["WS_F"].values,
        METEO_DAILY_DATA["EA"].values,
        METEO_DAILY_DATA["PA_F"].values,
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

    et = met.flux_2_evaporation(le, t_k=METEO_DAILY_DATA["TA_F"].values, time_domain=24)
    kcs_sw = et / METEO_DAILY_DATA["ET_ref"].values
    out_file = os.path.join(OUTPUT_FOLDER, "lai_vs_kc.csv")
    if not os.path.isdir(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    result = pd.DataFrame({"LAI": lais, "Kc": kcs_sw})
    result.to_csv(out_file, index=False)
    plot_kcs(METEO_DAILY_DATA["DATE"], lais, METEO_DAILY_DATA["ET_ref"],
             et, kcs_sw)


def lst_from_4sail(e_v, e_s, t_c, t_s, lais, vza, leaf_angle, t_atm=0):

    # Apply Kirchoff's law to get the soil and leaf bihemispherical reflectances
    rsoil = np.full((1, N_SIM), 1 - e_s)
    rho_leaf = np.full((1, N_SIM), 1 - e_v)
    tau_leaf = np.zeros((1, N_SIM))
    # Calculate the lidf,
    lidf = fs.calc_lidf_campbell_vec(np.full(N_SIM, leaf_angle))

    # 4SAIL for canopy reflectance and transmittance factors
    [tss, too, tsstoo, rdd, tdd, rsd, tsd, rdo, tdo, rso, rsos, rsod, rddt,
     rsdt, rdot,
     rsodt, rsost, rsot, gammasdf, gammasdb,
     gammaso] = fs.foursail_vec(lais,
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


