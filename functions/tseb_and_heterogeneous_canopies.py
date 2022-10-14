import os
from pyTSEB import TSEB
from pyTSEB import MO_similarity as mo
from pyTSEB import wind_profile as wind
from pyTSEB import resistances as res
from pyTSEB import energy_combination_ET as pet
from pyTSEB import meteo_utils as met
from pyTSEB import net_radiation as rad
from pyTSEB import clumping_index as ci
from pyTSEB import physiology as lf
from pypro4sail import machine_learning_regression as inv
from pypro4sail import prospect as pro
from pypro4sail import four_sail as sail
from model_evaluation import double_collocation as dc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import ipywidgets as w
import warnings

warnings.filterwarnings("ignore")

print("Gracias! librerías correctamente importadas")
print("Puedes continuar con las siguientes tareas")

slide_kwargs = {"continuous_update": False}
FIGSIZE = (12.0, 6.0)

np.seterr(all="ignore")
INPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
METEO_FILE_PATH = os.path.join(INPUT_FOLDER, "meteo",
                               "meteo_olive.csv")
METEO_DAILY_FILE_PATH = os.path.join(INPUT_FOLDER, "meteo",
                                     "meteo_daily_olive.csv")
LAI_FILE_PATH = os.path.join(INPUT_FOLDER, "canopy", "LAI_FAPAR_ES-Cnd.csv")
SOIL_FOLDER = os.path.join(INPUT_FOLDER, "soil_spectral_library")
ALBEDO_WEIGHT_FILE = os.path.join(INPUT_FOLDER, "albedo_weights.csv")

N_SIM = 50
pet.ITERATIONS = 5
LAI = 1.5
MIN_LAI = 0
MAX_LAI = 4
# Calculations are carried out for the following meteorological conditions: R, =
# 400 W m-2; D = 0,10,20 mb; T, = 25 “C; and u = 2ms-’. Such meteorological conditions
# might be considered typical for midday in the middle of a growing season at a subtropical
# site. However, the objective is not to make detailed predictions for particular meteoro-
# logical conditions, it is rather to illustrate the general features of the theoretical treatment
# described.
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
FLUX_LIMS = -100, 800
ET_LIMS = 0, 10
U = 5.  # set wind speed (measured at 10m above the canopy)
Z_U = 10.
Z_T = 10.
LAI_REF = 24 * 0.12
H_C_REF = 0.12  # Canopy height
LEAF_WIDTH = 0.02  # Leaf width (m)
# Create list of heights
ZS = np.linspace(0, Z_U, N_SIM)
US = np.linspace(0.50, 20, N_SIM)
EMIS_C = 0.98
EMIS_S = 0.94
RES_FORM = [TSEB.KUSTAS_NORMAN_1999, {"KN_c": 0.0038}]
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

LAT_CND = 37.914998
LON_CND = -3.227659
STDLON_CND = 15.
ELEV_CND = 366.0
E_SURF = 0.98

C_KC = [404.9, 79430.0]
C_KO = [278.4, 36380.0]
C_TES = [42.75, 37830.0]
C_VCX = [61.4, 65330., 149250., 485.]
C_JX = [np.exp(1.010 + 0.890 * np.log(C_VCX[0])),
        43540, 152040., 495.]
C_RD = [0.015 * C_VCX[0],
        46390.0, 150650.0, 490.]
C_TPU = (21.46, 53.100e3)
THETA = 0.9
ALPHA = 0.20
KN = 0.11
# Jmax decay
KD_STAR = 0.11
# mol H20 m-2 s-1
F_SOIL_10_0 = 1.5
D_0 = 10.
A_1 = 11.
G0P = 0.0001
CA = 412

C_D_MASSMAN = 0.2
OVERPASS_TIME = 10.75

LAI_DATA = pd.read_csv(LAI_FILE_PATH)
LAI_DATA["DATE"] = pd.to_datetime(LAI_DATA['DATE'], format="%Y-%m-%d").dt.date
METEO_DATA = pd.read_csv(METEO_FILE_PATH, na_values=-9999)
METEO_DATA['TIMESTAMP_START'] = pd.to_datetime(METEO_DATA['TIMESTAMP_START'],
                                               format="%Y%m%d%H%M")
METEO_DATA['TIMESTAMP_END'] = pd.to_datetime(METEO_DATA['TIMESTAMP_END'],
                                             format="%Y%m%d%H%M")
METEO_DATA['TIMESTAMP'] = METEO_DATA['TIMESTAMP_START'] + 0.5 * (
            METEO_DATA['TIMESTAMP_END'] -
            METEO_DATA['TIMESTAMP_START'])

METEO_DATA = METEO_DATA[["TIMESTAMP", "TA_F", "VPD_F", "SW_IN_F", "LW_IN_F", "SW_OUT",
                         "LW_OUT", "NETRAD", "WS_F", "PA_F", "LE_F_MDS", "H_F_MDS",
                         "G_F_MDS", "LE_CORR"]]

# Convert pressure units to mb
METEO_DATA["DOY"] = METEO_DATA["TIMESTAMP"].dt.day_of_year
METEO_DATA["TOD"] = METEO_DATA["TIMESTAMP"].dt.hour + METEO_DATA[
    "TIMESTAMP"].dt.minute / 60
METEO_DATA["VPD_F"] = 10 * METEO_DATA["VPD_F"]
METEO_DATA["PA_F"] = 10 * METEO_DATA["PA_F"]
METEO_DATA["TA_F"] = METEO_DATA["TA_F"] + 273.15
METEO_DATA["DATE"] = METEO_DATA["TIMESTAMP"].dt.date
E_SURF = 0.98
METEO_DATA["LE"] = METEO_DATA["NETRAD"] - METEO_DATA["G_F_MDS"] - METEO_DATA["H_F_MDS"]
METEO_DATA["LST"] = ((METEO_DATA['LW_OUT'] - (1. - E_SURF) * METEO_DATA['LW_IN_F']) / (
            rad.SB * E_SURF)) ** 0.25
METEO_DATA['SZA'] = met.calc_sun_angles(LAT_CND, LON_CND, STDLON_CND,
                                        METEO_DATA["DOY"], METEO_DATA["TOD"])[0]

METEO_DATA = METEO_DATA.merge(LAI_DATA, on="DATE")

METEO_DAILY_DATA = pd.read_csv(METEO_DAILY_FILE_PATH, na_values=-9999)
METEO_DAILY_DATA['DATE'] = pd.to_datetime(METEO_DAILY_DATA['TIMESTAMP'],
                                          format="%Y%m%d").dt.date
METEO_DAILY_DATA = METEO_DAILY_DATA[
    ["DATE", "TA_F", "VPD_F", "SW_IN_F", "LW_IN_F", "SW_OUT",
     "LW_OUT", "NETRAD", "WS_F", "PA_F", "LE_F_MDS", "H_F_MDS",
     "G_F_MDS"]]

# Convert pressure units to mb
METEO_DAILY_DATA["VPD_F"] = 10 * METEO_DAILY_DATA["VPD_F"]
METEO_DAILY_DATA["PA_F"] = 10 * METEO_DAILY_DATA["PA_F"]
METEO_DAILY_DATA["TA_F"] = METEO_DAILY_DATA["TA_F"] + 273.15
METEO_DAILY_DATA["LE"] = METEO_DAILY_DATA['NETRAD'] - METEO_DAILY_DATA['G_F_MDS'] \
                             - METEO_DAILY_DATA['H_F_MDS']

# METEO_DAILY_DATA["LE_BR"] = met.bowen_ratio_closure(METEO_DAILY_DATA["NETRAD"],
#                                                     METEO_DAILY_DATA["G_F_MDS"],
#                                                     METEO_DAILY_DATA["H_F_MDS"],
#                                                     METEO_DAILY_DATA["LE_F_MDS"])[0]
#
# METEO_DAILY_DATA["LE"] = np.nanmean([METEO_DAILY_DATA["LE_RES"], METEO_DAILY_DATA["LE_BR"],
#                                      METEO_DAILY_DATA["LE_F_MDS"]], axis=0)

METEO_DAILY_DATA["ET"] = met.flux_2_evaporation(METEO_DAILY_DATA["LE"],
                                                METEO_DAILY_DATA["TA_F"],
                                                24)

METEO_DAILY_DATA = METEO_DAILY_DATA.merge(LAI_DATA, on="DATE")

# METEO_DAILY_DATA = METEO_DATA.groupby("DATE").mean()
# METEO_DAILY_DATA["ET"] = met.flux_2_evaporation(METEO_DAILY_DATA["LE"],
#                                                 METEO_DAILY_DATA["TA_F"],
#                                                 24)
# METEO_DAILY_DATA.reset_index(inplace=True)
# METEO_DAILY_DATA = METEO_DAILY_DATA.rename(columns={'index': 'DATE'})

w_kappa = w.FloatSlider(min=0, max=1, value=0.5,
                        description='$\kappa$ (--)',
                        description_tooltip="Coeficiente de extinción",
                        **slide_kwargs)
w_cab = w.FloatSlider(value=inv.MEAN_CAB,
                      min=inv.MIN_CAB,
                      max=inv.MAX_CAB,
                      step=1, description='Cab ($\mu$g/cm²)',
                      description_tooltip="Concentración de clorofila",
                      **slide_kwargs)
w_lai = w.FloatSlider(value=inv.MEAN_LAI,
                      min=MIN_LAI,
                      max=MAX_LAI,
                      step=0.1, description='LAI (m²/m²)',
                      description_tooltip="Índice de Área Foliar del ecosistema",
                      **slide_kwargs)
w_leaf_angle = w.FloatSlider(value=57, min=1, max=90, step=1,
                             description='LIDF (deg.)',
                             description_tooltip="Ángulo zenital predominante de las hojas",
                             **slide_kwargs)
w_sza = w.FloatSlider(value=37, min=0, max=89, step=1,
                      description='SZA (deg.)',
                      description_tooltip="Ángulo zenital solar",
                      **slide_kwargs)
w_saa = w.FloatSlider(value=180, min=0, max=359, step=1,
                      description='SAA (deg.)',
                      description_tooltip="Ángulo azimutal solar",
                      **slide_kwargs)
w_hc = w.FloatSlider(min=0.01, max=8, value=4, step=0.01, description='$h_c$ (m)',
                     description_tooltip="Altura del dosel",
                     **slide_kwargs)
w_hb_ratio = w.FloatSlider(min=0.0, max=0.9, value=0.5, step=0.01,
                           description="$h_{bottom}$ (--)",
                           description_tooltip='Altura relativa donde se encuentra el inicio del dosel',
                           **slide_kwargs)

w_h_c_max = w.FloatSlider(min=0.1, max=0.9, value=0.5, step=0.01,
                          description='$h_{max}$ (--)',
                          description_tooltip='Posición relativa dentro del dosel donde se encuentre el máximo de biomasa',
                          **slide_kwargs)

w_wc = w.FloatSlider(min=0.1, max=3, value=1, step=0.01, description='Forma',
                     description_tooltip="Factor de forma, estimnado como el ratio entre el ancho y el alto de copa",
                     **slide_kwargs)
w_fc = w.FloatSlider(min=0.05, max=1, value=0.2, step=0.01,
                     description='$f_c$ (--)',
                     description_tooltip='Porcentaje de ocupación del suelo',
                     **slide_kwargs)
w_interrow = w.FloatSlider(min=0.1, max=6, value=1, step=0.1,
                           description='L (m)',
                           description_tooltip='Separación entre hileras',
                           **slide_kwargs)
w_psi = w.FloatSlider(min=-90., max=90, value=0, description='Orientación (deg.)',
                      description_tooltip='Orientación azimutal de las hileras',
                      **slide_kwargs)

w_skyl = w.FloatSlider(min=0, max=1, value=0.1, step=0.01,
                       description='Rad. difusa (--)',
                       description_tooltip='Porcentaje de radiaciación difusa',
                       **slide_kwargs)
w_leaf_abs = w.FloatSlider(min=0.8, max=1, value=0.9, step=0.01,
                           description='Abs. hoja',
                           description_tooltip='Absortividad de la hoja',
                           **slide_kwargs)

w_tair = w.FloatSlider(min=273, max=313, value=293, step=0.1,
                       description='T$_{aire}$ (K)',
                       description_tooltip="Temperatura del aire",
                       **slide_kwargs)
w_hr = w.FloatSlider(min=0, max=100, value=50, step=0.1,
                     description='HR (%)',
                     description_tooltip="Humedad relativa del aire",
                     **slide_kwargs)
w_emiss = w.FloatSlider(min=0.9, max=0.99, value=0.98, step=0.01,
                        description='$\epsilon$',
                        description_tooltip="Emisividad de la superficie",
                        **slide_kwargs)
w_lat = w.FloatSlider(min=0.0, max=90, value=40, step=1,
                      description='Latitud (deg.)',
                      description_tooltip="Latitud de la parcela",
                      **slide_kwargs)

w_zol = w.FloatSlider(min=-1, max=1, value=0, step=0.01, description='Estabilidad',
                      description_tooltip="Coeficiente de estabilidad atmosférica",
                      **slide_kwargs)
w_u = w.FloatSlider(min=0.1, max=20, value=U, step=0.1, description='WS (m/s)',
                    description_tooltip="Velocidad del viento",
                    **slide_kwargs)

w_r_ss = w.FloatSlider(min=0, max=10000, value=2000, step=100,
                       description='R$_{ss}$ (s/m)',
                       description_tooltip="Resistencia al tranposrte de humedad en el suelo\n"
                                           "Valores cercanos a cero indican suelos más encharcados\n"
                                           "o mayor contenido de agua en la capa superficial del suelo",
                                           **slide_kwargs)

w_g_st = w.FloatSlider(min=0, max=0.5, value=GST_REF, step=0.001,
                       description='g$_{st}$ (mmol/m²s¹)',
                       description_tooltip="Conductancia estomática de la hoja\n"
                                           "Valores cercanos a cero indican más estrés hídrico\n"
                                           "o menor contenido de agua en la zona radicular del suelo",
                       **slide_kwargs)
w_vza = w.FloatSlider(min=0, max=89, value=0, step=1,
                      description='VZA (deg.)',
                      description_tooltip="Ángulo cenital de observación",
                      **slide_kwargs)
w_ev = w.FloatSlider(min=0.97, max=1, value=0.99, step=0.001,
                     description='$\epsilon_V$',
                     description_tooltip="Emisividad de la vegetación",
                     readout_format='.3f',
                     **slide_kwargs)
w_es = w.FloatSlider(min=0.90, max=1, value=0.97, step=0.001,
                     description='$\epsilon_S$',
                     description_tooltip="Emisividad del suelo",
                     readout_format='.3f',
                     **slide_kwargs)

dates = pd.date_range(METEO_DATA["DATE"].iloc[0], METEO_DATA["DATE"].iloc[-1], freq='D')
date_opts = options = [(date.strftime('%d %b %Y'), date) for date in dates]
w_dates = w.SelectionRangeSlider(options=date_opts,
                                 index=(0, len(date_opts) - 1),
                                 description='Fechas',
                                 description_tooltip="Rango de fechas para mostrar en la serie temporal",
                                 orientation='horizontal',
                                 layout={'width': '500px'},
                                 **slide_kwargs)

w_stress = w.FloatSlider(min=0, max=1, value=0.5, step=0.01,
                         description='Stress (--)',
                         description_tooltip='Factor de estrés hídrico\n'
                                             '0: Sin estres, disponibilidad hídrica total de la planta\n'
                                             '1: Planta totalmente estresada, bajo el umbral del punto de marchitez',
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


def plot_net_solar_radiation(lai, leaf_angle, h_c, f_c, row_distance,
                             row_direction, skyl=0.1, fvis=0.55,
                             lat=LATITUDE):
    chi = rad.leafangle_2_chi(leaf_angle)
    local_lai = np.full(N_SIM, lai) / f_c
    szas, saas = met.calc_sun_angles(np.full(N_SIM, lat),
                                     np.zeros(N_SIM),
                                     np.zeros(N_SIM),
                                     np.full(N_SIM, 180),
                                     HOURS)
    sundown = szas > 89
    sdn_dir = (1. - skyl)
    sdn_dif = skyl
    w_c = row_distance * f_c / np.full(N_SIM, h_c)
    psi = relative_azimuth(row_direction, saas)
    omega = ci.calc_omega_rows(np.full(N_SIM, lai),
                               np.full(N_SIM, f_c),
                               theta=szas,
                               psi=psi,
                               w_c=w_c,
                               x_lad=np.full(N_SIM, chi))
    lai_eff = local_lai * omega
    # Compute radiation for row crop canopy
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
    sn_v[sundown] = np.nan
    sn_s[sundown] = np.nan
    # Compute radiation for homogeneous canopy
    sn_v_0, sn_s_0 = rad.calc_Sn_Campbell(lai,
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
                                          LAI_eff=None)
    sn_v_0[sundown] = np.nan
    sn_s_0[sundown] = np.nan
    sn = sn_v + sn_s
    sn_0 = sn_v_0 + sn_s_0
    fig, axs = plt.subplots(figsize=FIGSIZE)
    # Plot the results
    axs.plot(HOURS, sn, 'black', label="Net Radiation row crop")
    axs.plot(HOURS, sn_0, 'black', ls="--", label="Net Radiation homogenous crop")
    axs.plot(HOURS, sn_v, 'green', label="Net Canopy Radiation row crop")
    axs.plot(HOURS, sn_v_0, 'green', ls="--",
             label="Net Canopy Radiation homogenous crop")
    # axs.plot(HOURS, sn_s, 'orange', label="Net Soil Radiation row crop")
    # axs.plot(HOURS, sn_s_0, 'orange', ls="--", label="Net Soil Radiation homogenous crop")
    axs.set_xlabel('Time (hours)')
    axs.set_ylabel('Fraction of absorbed radiation (--)')
    axs.set_xlim((TIME_START, TIME_END))
    axs.set_ylim((0, 1))
    axs.legend()
    # secax = axs.twinx()
    # secax.plot(HOURS, albedo, "b")
    # secax.plot(HOURS, albedo_0, "b", ls="--")
    # secax.set_ylabel('Albedo', color="blue")
    # secax.tick_params(axis='y', colors='blue')
    # secax.set_ylim((0, 1))
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


def wind_profile_homogeneous(zol, lai, h_c=0.12):
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


def wind_profile_heterogeneous(zol, lai, h_c, hb_ratio=0.5, h_c_max=0.5):
    u_z_0, _ = wind_profile_homogeneous(zol, lai, h_c=h_c)
    u_z_ref, _ = wind_profile_homogeneous(0., LAI_REF, h_c=H_C_REF)

    # Estimate surface roughness
    z_0m, d_0 = calc_roughness(h_c)
    l_mo = l_2_zol(z_0m, zol)
    # Calculate the friction velocity
    u_friction = mo.calc_u_star(U, Z_U, l_mo, d_0, z_0m)
    upper = ZS >= h_c
    u_z = np.ones(ZS.shape)
    u_c = wind.calc_u_C_star(u_friction, h_c, d_0, z_0m, l_mo)
    u_z[upper] = wind.calc_u_C_star(u_friction, ZS[upper], d_0, z_0m, l_mo)
    h_b = hb_ratio * h_c
    Xi_max, sigma_u, sigma_l = wind.canopy_shape(h_c,
                                                 h_b=h_b,
                                                 h_max=h_c_max)
    f_a = wind.calc_canopy_distribution(Xi_max, sigma_u, sigma_l)
    f_a_cum = wind.calc_cummulative_canopy_distribution(f_a)
    u_z[~upper] = wind.calc_u_Massman(np.full(np.sum(~upper), u_c),
                                      np.full(np.sum(~upper), h_c),
                                      np.full(np.sum(~upper), lai),
                                      ZS[~upper],
                                      f_a_cum,
                                      xi_soil=Z0_SOIL / h_c)

    # Plots the wind profile for given stability lenght compared to a neutral atmosphere
    fig, axs = plt.subplots(ncols=2, figsize=FIGSIZE, sharey=True)
    axs[0].plot(f_a, np.linspace(0, h_c, np.size(f_a)))
    axs[0].set_ylim((0, h_c))
    axs[0].set_xlim((0, None))
    axs[0].set_xlabel('Foliar density')
    axs[0].set_ylabel('Height above ground (m)')

    axs[1].plot(u_z, ZS, 'b', label="Wind profile heterogeneous")
    axs[1].plot(u_z_0, ZS, 'b--', label="Wind profile homogeneous")
    # Plot the ufriction wind canopy
    axs[1].plot(u_c, h_c, marker='*', markerfacecolor="none", markeredgecolor="blue",
                ms=12, ls="None", label='$u_c$')

    axs[1].plot(u_z_ref, ZS, 'k--', label="FAO56 reference profile")
    # Plot the canopy windspeed according to the two different methods
    axs[1].legend(loc='upper left')
    axs[1].set_xlim((0, U))
    axs[1].set_ylim((0, Z_U))
    axs[1].set_xlabel('Wind Speed (m)')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()

    return u_z, u_c


def plot_aerodynamic_resistance(zol, h_c):
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
    ra = calc_r_a(zol, h_c)
    axs.plot(US, ra, 'k')
    axs.set_ylabel('Aerodynamic Resistance (s/m)')
    axs.set_ylim((0, 200))
    axs.set_xlabel('Wind speed (m/s)')
    axs.set_xlim((0, None))
    plt.tight_layout()
    plt.show()


def plot_flux_variation(values, le, le_c, le_pm, le_fao, t_c, t_s, t_0,
                        var="LAI"):
    fig, axs = plt.subplots(3, figsize=FIGSIZE, sharex=True)
    axs[0].plot(values, le, linestyle="-", color="blue", label="ET$_{SW}$")
    axs[0].plot(values, le_pm, linestyle="-", color="red", label="ET$_{PM}$")
    axs[0].plot(0.5 * LAI_REF, le_fao, color="black", markersize=12, marker="*",
                ls="none",
                label="ET$_{FAO56}$")

    axs[1].plot(values, le_c / le, linestyle="-", color="green", label="Canopy")
    axs[1].plot(values, 1 - le_c / le, linestyle="-", color="orange", label="Soil")

    axs[2].plot(values, t_0, linestyle="-", color="black", label="T$_0$ - T$_a$")
    axs[2].plot(values, t_c, linestyle="-", color="green", label="T$_c$ - T$_a$")
    axs[2].plot(values, t_s, linestyle="-", color="orange", label="T$_s$ - T$_a$")
    axs[2].axhline(0, c="silver", ls=":")

    value_lims = np.min(values), np.max(values)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[0].set_ylabel("ET (mm / day)$)")
    axs[0].set_ylim(ET_LIMS)
    axs[1].set_ylabel("Fraction of ET")
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
        resistance_form=[TSEB.KUSTAS_NORMAN_1999, {"KN_c": np.full(N_SIM, 0.0038)}],
        calcG_params=[[1], 0.35],
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
                                                 calcG_params=[[1],
                                                               0.35],
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
                        t_c - tair, t_s - tair, t_0 - tair,
                        var="LAI")
    return t_c, t_s, t_0


def get_land_surface_temperature(vza, leaf_angle, temperatures, e_v=0.98, e_s=0.95):
    t_c, t_s, t_0 = temperatures.result
    bt_obs, emiss = lst_from_4sail(e_v, e_s, t_c, t_s, LAIS, vza, leaf_angle, t_atm=243.)
    chi = rad.leafangle_2_chi(leaf_angle)
    fc = TSEB.calc_F_theta_campbell(vza, LAIS, x_LAD=chi)
    lst = (fc * t_c ** 4 + (1. - fc) * t_s ** 4) ** 0.25
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
    lw = (rdot * h_sky + (aeev * h_vc + aees * h_gc)) / np.pi
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


def prepare_tseb_data(h_c, f_c, w_c, hb_ratio, h_c_max, x_lad, set_stress=None):
    dims = len(METEO_DATA.index)
    omega_0 = TSEB.CI.calc_omega0_Kustas(METEO_DATA["LAI"].values, f_c, x_LAD=x_lad,
                                         isLAIeff=True)
    omega = TSEB.CI.calc_omega_Kustas(omega_0, METEO_DATA["SZA"].values, w_C=w_c)

    local_lai = METEO_DATA["LAI"] / f_c
    lai_eff = local_lai * omega
    fc0 = TSEB.calc_F_theta_campbell(0, local_lai, w_c, omega_0, x_lad)
    # Estimates the direct and diffuse solar radiation
    difvis, difnir, fvis, fnir = TSEB.rad.calc_difuse_ratio(METEO_DATA["SW_IN_F"],
                                                            np.minimum(METEO_DATA["SZA"],
                                                                       90),
                                                            press=METEO_DATA["PA_F"])

    skyl = fvis * difvis + fnir * difnir
    sdn_dir = (1. - skyl) * METEO_DATA["SW_IN_F"]
    sdn_dif = skyl * METEO_DATA["SW_IN_F"]
    sn_c, sn_s = TSEB.rad.calc_Sn_Campbell(
        METEO_DATA["LAI"].values,
        METEO_DATA["SZA"].values,
        sdn_dir,
        sdn_dif,
        fvis,
        fnir,
        np.full(dims, 0.07),
        np.full(dims, 0.07),
        np.full(dims, 0.35),
        np.full(dims, 0.35),
        np.full(dims, 0.15),
        np.full(dims, 0.25),
        x_LAD=np.full(dims, x_lad),
        LAI_eff=lai_eff)

    sn_c[~np.isfinite(sn_c)] = 0
    sn_s[~np.isfinite(sn_s)] = 0

    # Calculate roughness for momentum (zo)
    # and zero-plane displacement height (d)
    [z_0m, d_0] = TSEB.res.calc_roughness(METEO_DATA["LAI"].values,
                                          np.full(dims, h_c),
                                          np.full(dims, w_c),
                                          np.full(dims, res.BROADLEAVED_E),
                                          f_c=np.full(dims, f_c))

    d_0[d_0 < 0] = 0
    z_0m[np.isnan(z_0m)] = Z0_SOIL
    z_0m[z_0m < Z0_SOIL] = Z0_SOIL
    h_b = hb_ratio * h_c
    Xi_max, sigma_u, sigma_l = wind.canopy_shape(h_c,
                                                 h_b=h_b,
                                                 h_max=h_c_max)
    f_a = wind.calc_canopy_distribution(Xi_max, sigma_u, sigma_l)
    f_a_cum = wind.calc_cummulative_canopy_distribution(f_a)
    r_st = None
    fc0 = None
    if type(set_stress) != type(None):
        fc0 = TSEB.calc_F_theta_campbell(np.zeros(dims), METEO_DATA["LAI"],
                                         w_C=w_c, Omega0=omega_0, x_LAD=x_lad)
        par_in = fvis * METEO_DATA["SW_IN_F"].values / lf.MUEINSTEIN_2_WATT
        par_dir = (1. - skyl) * par_in
        par_dif = skyl * par_in

        # Initialize potential stomatal resistance
        l_mo = np.inf
        z_oh = TSEB.res.calc_z_0H(z_0m)
        u_friction = TSEB.MO.calc_u_star(METEO_DATA["WS_F"].values, 9, l_mo, d_0, z_0m)
        r_a_params = {"z_T": Z_T, " u_friction": u_friction, "L": l_mo,
                      "d_0": d_0, "z_0H": z_oh}
        r_x_params = {"u_friction": u_friction,
                      "h_C": np.full(dims, h_c),
                      "d_0": d_0,
                      "z_0M": z_0m,
                      "L": l_mo,
                      "F": local_lai,
                      "LAI": METEO_DATA["LAI"].values,
                      "leaf_width": LEAF_WIDTH,
                      "res_params": {"KN_c": 0.0038},
                      "massman_profile": [C_D_MASSMAN, f_a_cum],
                      "z0_soil": np.full(dims, Z0_SOIL)}

        r_a, r_x, *_ = TSEB.calc_resistances(TSEB.KUSTAS_NORMAN_1999,
                                             {"R_x": r_x_params,
                                              "R_a": r_a_params})

        apar = (1. - 0.07) * par_in
        g_st_0 = lf.gpp_leaf_no_gs(METEO_DATA["TA_F"].values,
                                   METEO_DATA["VPD_F"].values,
                                   apar,
                                   ca=CA,
                                   theta=THETA,
                                   alpha=ALPHA,
                                   c_kc=C_KC,
                                   c_ko=C_KO,
                                   c_tes=C_TES,
                                   c_rd=C_RD,
                                   c_vcx=C_VCX,
                                   c_jx=C_JX,
                                   g0p=G0P,
                                   a_1=A_1,
                                   d_0=D_0,
                                   fw=np.full(dims, 1 - set_stress),
                                   verbose=False)[2]

        # r_soil_0 = lf.soil_respiration_lloyd(METEO_DATA["TA_F"].values,
        #                                      f_soil_ref=F_SOIL_10_0)
        # _, _, g_st_0, *_ = lf.gpp_canopy_no_gs(METEO_DATA["VPD_F"].values,
        #                                        r_x,
        #                                        r_a,
        #                                        METEO_DATA["TA_F"].values,
        #                                        METEO_DATA["LAI"].values,
        #                                        par_dir,
        #                                        par_dif,
        #                                        METEO_DATA["SZA"].values,
        #                                        lai_eff=lai_eff,
        #                                        x_lad=x_lad,
        #                                        rho_leaf=np.full(dims, 0.07),
        #                                        tau_leaf=np.full(dims, 0.07),
        #                                        rho_soil=np.full(dims, 0.15),
        #                                        press=METEO_DATA["PA_F"].values,
        #                                        ca=CA,
        #                                        theta=THETA,
        #                                        alpha=ALPHA,
        #                                        c_kc=C_KC,
        #                                        c_ko=C_KO,
        #                                        c_tes=C_TES,
        #                                        c_rd=C_RD,
        #                                        c_vcx=C_VCX,
        #                                        c_jx=C_JX,
        #                                        f_soil=r_soil_0,
        #                                        kn=KN,
        #                                        kd_star=KD_STAR,
        #                                        g0p=G0P,
        #                                        a_1=A_1,
        #                                        d_0=D_0,
        #                                        fw=np.full(dims, 1 - set_stress),
        #                                        leaf_type=1,
        #                                        verbose=False)

        g_st_0[g_st_0 < G0P] = G0P
        g_st_0 *= lf.GV_GC_RATIO
        daytime = METEO_DATA["SW_IN_F"] > 50
        no_valid = np.logical_and(~np.isfinite(g_st_0), ~daytime)
        g_st_0[no_valid] = G0P * lf.GV_GC_RATIO

        r_st = 1. / (TSEB.res.molm2s1_2_ms1(METEO_DATA["TA_F"].values,
                                            METEO_DATA["PA_F"].values) * g_st_0)
        r_st[r_st < 0] = np.nan

    return dims, sn_c, sn_s, f_a_cum, z_0m, d_0, r_st, fc0


def simulate_flux_timeseries(h_c, f_c, w_c, hb_ratio, h_c_max, leaf_angle, stress, r_ss,
                             date_range):
    print("Simulando flujos, espera un momento...")
    x_lad = rad.leafangle_2_chi(leaf_angle)
    dims, sn_c, sn_s, f_a_cum, z_0m, d_0, r_st, fc0 = prepare_tseb_data(h_c,
                                                                        f_c,
                                                                        w_c,
                                                                        hb_ratio,
                                                                        h_c_max,
                                                                        x_lad,
                                                                        set_stress=stress)

    ea = TSEB.met.calc_vapor_pressure(METEO_DATA["TA_F"].values) - METEO_DATA[
        "VPD_F"].values

    [flag, t_s, t_c, _, ln_s, ln_c, le, h, le_c, _, le_s, *_] = pet.shuttleworth_wallace(
        METEO_DATA["TA_F"].values,
        METEO_DATA["WS_F"].values,
        ea,
        METEO_DATA["PA_F"].values,
        sn_c,
        sn_s,
        METEO_DATA["LW_IN_F"].values,
        METEO_DATA["LAI"].values,
        np.full(dims, h_c),
        EMIS_C,
        EMIS_S,
        z_0m,
        d_0,
        np.full(dims, 9),
        9,
        leaf_width=LEAF_WIDTH,
        z0_soil=Z0_SOIL,
        Rst_min=r_st,
        R_ss=r_ss,
        resistance_form=[TSEB.KUSTAS_NORMAN_1999, {"KN_c": np.full(dims, 0.0038)}],
        calcG_params=[[1], 0.35],
        leaf_type=1,
        massman_profile=[C_D_MASSMAN, f_a_cum],
        verbose=False)

    le = np.clip(le, 0, FLUX_LIMS[1])
    no_valid = flag == 255
    le[no_valid] = np.nan
    t_c[no_valid] = np.nan
    t_s[no_valid] = np.nan
    out_df = METEO_DATA[["TIMESTAMP", "DATE", "LAI", "TA_F"]]
    out_df["LE"] = le
    out_df["NETRAD"] = sn_c + sn_s + ln_c + ln_s

    out_df["LST"] = ((fc0 * met.calc_stephan_boltzmann(t_c)
                      + (1 - fc0) * met.calc_stephan_boltzmann(t_s)) / rad.SB) ** 0.25

    daily_df = out_df[["DATE", "LAI", "TA_F", "LE"]].groupby("DATE").mean()

    daily_df["ET"] = met.flux_2_evaporation(daily_df["LE"],
                                            t_k=daily_df["TA_F"],
                                            time_domain=24)
    daily_df.reset_index(inplace=True)
    daily_df = daily_df.rename(columns={'index': 'DATE'})
    print("¡Hecho!")
    plot_flux_timeseries(date_range, out_df, daily_df, include_obs=True)

    return out_df, daily_df


def run_tseb(h_c, f_c, w_c, hb_ratio, h_c_max, leaf_angle):
    print("Estimando flujos, espera un momento...")
    x_lad = rad.leafangle_2_chi(leaf_angle)
    dims, sn_c, sn_s, f_a_cum, z_0m, d_0, *_ = prepare_tseb_data(h_c,
                                                                 f_c,
                                                                 w_c,
                                                                 hb_ratio,
                                                                 h_c_max,
                                                                 x_lad,
                                                                 set_stress=None)

    ea = TSEB.met.calc_vapor_pressure(METEO_DATA["TA_F"].values) - METEO_DATA[
        "VPD_F"].values

    [flag, t_s, t_c, _, ln_s, ln_c, le_c, h_c, le_s, h_s, *_] = TSEB.TSEB_PT(
        METEO_DATA["LST"].values,
        np.zeros(dims),
        METEO_DATA["TA_F"].values,
        METEO_DATA["WS_F"].values,
        ea,
        METEO_DATA["PA_F"].values,
        sn_c,
        sn_s,
        METEO_DATA["LW_IN_F"].values,
        METEO_DATA["LAI"].values,
        np.full(dims, h_c),
        0.99,
        EMIS_S,
        z_0m,
        d_0,
        np.full(dims, 9),
        9,
        x_LAD=x_lad,
        f_c=f_c,
        w_C=w_c,
        leaf_width=LEAF_WIDTH,
        z0_soil=Z0_SOIL,
        resistance_form=[TSEB.KUSTAS_NORMAN_1999, {"KN_c": np.full(dims, 0.0038)}],
        calcG_params=[[1], 0.35],
        verbose=False
        # massman_profile=[C_D_MASSMAN, f_a_cum]
    )

    le = np.clip(le_c + le_s, -100, 500)
    no_valid = flag == 255
    le[no_valid] = np.nan
    out_df = METEO_DATA[["TIMESTAMP", "DATE", "TOD", "LAI", "TA_F", "SW_IN_F", "LST"]]
    out_df["LE"] = le
    out_df["TSEB_QC"] = flag
    out_df["NETRAD"] = sn_c + sn_s + ln_c + ln_s

    daily_df = out_df[["DATE", "LAI", "TA_F", "SW_IN_F", "LE"]].groupby("DATE").mean()
    daily_df["ET"] = met.flux_2_evaporation(daily_df["LE"],
                                            t_k=daily_df["TA_F"],
                                            time_domain=24)
    daily_df.reset_index(inplace=True)
    daily_df = daily_df.rename(columns={'index': 'DATE'})

    print("¡Hecho!")

    validate_fluxes(out_df, daily_df)
    return out_df, daily_df


def validate_fluxes(out_df, daily_df):
    flux_lims = (-100, 500)
    et_lims = (0, 6)
    dates = np.logical_and.reduce((out_df["TSEB_QC"] < 5,
                                   METEO_DATA["SW_IN_F"] > 50,
                                   np.isfinite(out_df["LE"]),
                                   METEO_DATA["LE"] <= flux_lims[1],
                                   out_df["LE"] <= flux_lims[1],
                                   out_df["LE"] >= flux_lims[0],
                                   METEO_DATA["LE"] >= flux_lims[0]))

    fig, axs = plt.subplots(ncols=2, figsize=FIGSIZE)
    axs[0].scatter(out_df["LE"].loc[dates], METEO_DATA["LE"].loc[dates],
                   c='b', marker='.', alpha=0.2, label='H', s=3)
    axs[0].set_xlim(flux_lims)
    axs[0].set_ylim(flux_lims)
    axs[0].set_xlabel(r'Estimated (W/m²)')
    axs[0].set_ylabel(r'Observed (W/m²)')
    axs[0].set_title('Instantaneous Latent Heat Flux')
    axs[0].plot(flux_lims, flux_lims, 'k-')
    axs[0].grid()
    bias_le, mae_le, rmse_le = dc.error_metrics(METEO_DATA["LE"].loc[dates],
                                                out_df["LE"].loc[dates])

    cor_le, *_ = dc.agreement_metrics(METEO_DATA["LE"].loc[dates],
                                      out_df["LE"].loc[dates])

    axs[0].text(0.1, 0.9,
                f'RMSE = {rmse_le:>6.0f}\n'
                f'bias = {bias_le:>6.0f}\n'
                f'r:   = {cor_le:>6.2f}',
                backgroundcolor='white',
                family='monospace',
                linespacing=1,
                verticalalignment="top",
                transform=axs[0].transAxes
                )

    dates = np.logical_and.reduce((np.isfinite(METEO_DAILY_DATA["ET"]),
                                   np.isfinite(daily_df["ET"]),
                                   daily_df["ET"] <= et_lims[1],
                                   daily_df["ET"] >= et_lims[0],
                                   METEO_DAILY_DATA["ET"] <= et_lims[1],
                                   METEO_DAILY_DATA["ET"] >= et_lims[0]))

    axs[1].scatter(daily_df["ET"].loc[dates], METEO_DAILY_DATA['ET'].loc[dates],
                   c='b', marker='o', label='H', s=6)
    axs[1].set_xlim(et_lims)
    axs[1].set_ylim(et_lims)
    axs[1].set_xlabel(r'Estimated (mm/day)')
    axs[1].set_ylabel(r'Observed (mm/day)')
    axs[1].set_title('Daily ET')
    axs[1].plot(et_lims, et_lims, 'k-')
    axs[1].grid()
    bias_le, mae_le, rmse_le = dc.error_metrics(METEO_DAILY_DATA["ET"].loc[dates],
                                                daily_df["ET"].loc[dates])

    cor_le, *_ = dc.agreement_metrics(METEO_DAILY_DATA["ET"].loc[dates],
                                      daily_df["ET"].loc[dates])

    axs[1].text(0.1, 0.9,
                f'RMSE = {rmse_le:>6.2f}\n'
                f'bias = {bias_le:>6.2f}\n'
                f'r:   = {cor_le:>6.2f}',
                backgroundcolor='white',
                family='monospace',
                linespacing=1,
                verticalalignment="top",
                transform=axs[1].transAxes
                )

    plt.tight_layout()
    plt.show()

def create_flux_timeseries(date_range,
                           df=METEO_DATA,
                           daily_df=METEO_DAILY_DATA,
                           include_obs=False):

    plot_flux_timeseries(date_range, df, daily_df, include_obs=include_obs)


def plot_flux_timeseries(date_range, df, daily_df, include_obs=False):

    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=FIGSIZE)
    axs[0].plot(df["TIMESTAMP"], df["NETRAD"], "k:", label="$R n$")
    axs[0].plot(df["TIMESTAMP"], df["LE"], "b", label="$\lambda E$")
    axs[0].set_ylim(FLUX_LIMS)
    axs[0].set_ylabel("Flux (W/m²)")
    axs[0].legend()
    axs[1].plot(df["TIMESTAMP"], df["TA_F"] - 273.15, "k:", label="$T_{air}$")
    axs[1].plot(df["TIMESTAMP"], df["LST"] - 273.15, "r", label="LST")
    axs[1].legend()
    axs[1].set_ylim((0, 50))
    axs[1].set_ylabel("Temp. (C)")
    axs[2].plot(daily_df["DATE"], daily_df["LAI"], "g", label="LAI")
    axs[2].set_ylim((0, MAX_LAI))
    axs[2].set_ylabel("LAI")
    axs[2].set_xlim((date_range))
    axs[2].legend(loc="upper left")
    secax = axs[2].twinx()
    secax.plot(daily_df["DATE"], daily_df["ET"], "b", label="ET")
    if include_obs:
        secax.plot(METEO_DAILY_DATA["DATE"], METEO_DAILY_DATA["ET"], "b:",
                   label="Measured ET")

    secax.set_ylabel('ET')
    secax.set_ylim((0, 6))
    secax.legend(loc="upper right")
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%y"))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()
