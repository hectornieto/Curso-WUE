import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypro4sail import pypro4sail as fs
from pypro4sail import machine_learning_regression as inv
from pypro4sail import four_sail as sail
from pypro4sail import prospect as pro
import ipywidgets as w
from IPython.display import display, clear_output
print("Gracias! librerías correctamente importadas")
print("Puedes continuar con las siguientes tareas")

slide_kwargs = {"continuous_update": False}
FIGSIZE = (12.0, 6.0)

REGIONS = {"B": (400, 500), "G": (500, 600), "R": (600, 700),
               "R-E": (700, 800), "NIR": (800, 1100), "SWIR": (1100, 2500)}
# Generate the list with VZAs (from 0 to 89)
VZAS = np.arange(0, 90)
INPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
SOIL_FOLDER = os.path.join(INPUT_FOLDER, "soil_spectral_library")
SRF_FOLDER = os.path.join(INPUT_FOLDER, "sensor_response_functions")

PARAM_DICT = {"N_leaf": inv.MEAN_N_LEAF,
              "Cab": inv.MEAN_CAB,
              "Car": inv.MEAN_CAR,
              "Cbrown": inv.MEAN_CBROWN,
              "Ant": inv.MEAN_ANT,
              "Cm": inv.MEAN_CM,
              "Cw": inv.MEAN_CW,
              "LAI": inv.MEAN_LAI,
              "leaf_angle": inv.MEAN_LEAF_ANGLE,
              "hotspot": inv.MEAN_HOTSPOT,
              "SZA": 35.,
              "VZA": 0.,
              "PSI": 0.,
              "skyl": 0.2,
              "soil": 'ProSAIL_DrySoil.txt'}

RANGE_DICT = {"N_leaf": (inv.MIN_N_LEAF, inv.MAX_N_LEAF),
              "Cab": (inv.MIN_CAB, inv.MAX_CAB),
              "Car": (inv.MIN_CAR, inv.MAX_CAR),
              "Cbrown": (inv.MIN_CBROWN, inv.MAX_CBROWN),
              "Ant": (inv.MIN_ANT, inv.MAX_ANT),
              "Cm": (inv.MIN_CM, inv.MAX_CM),
              "Cw": (inv.MIN_CW, inv.MAX_CW),
              "LAI": (inv.MIN_LAI, inv.MAX_LAI),
              "leaf_angle": (inv.MIN_LEAF_ANGLE, inv.MAX_LEAF_ANGLE),
              "hotspot": (0, 1),
              "SZA": (0, 89),
              "PSI": (0, 180),
              "VZA": (0, 89),
              "skyl": (0, 1)}

N_STEPS = 10

soil_files = sorted(glob(os.path.join(SOIL_FOLDER, "*.txt")))
soil_types = [os.path.splitext(os.path.basename(i))[0]
                         for i in soil_files]

w_nleaf = w.FloatSlider(value=inv.MEAN_N_LEAF,
                        min=inv.MIN_N_LEAF,
                        max=inv.MAX_N_LEAF,
                        step=0.01, description='N:', **slide_kwargs)
w_cab = w.FloatSlider(value=inv.MEAN_CAB,
                      min=inv.MIN_CAB,
                      max=inv.MAX_CAB,
                      step=1, description='Cab ($\mu$g/cm²):', **slide_kwargs)
w_car = w.FloatSlider(value=inv.MEAN_CAR,
                      min=inv.MIN_CAR,
                      max=inv.MAX_CAR,
                      step=1, description='Car ($\mu$g/cm²):', **slide_kwargs)
w_ant = w.FloatSlider(value=inv.MEAN_ANT,
                      min=inv.MIN_ANT,
                      max=inv.MAX_ANT,
                      step=1, description='Ant ($\mu$g/cm²):', **slide_kwargs)
w_cbrown = w.FloatSlider(value=inv.MEAN_CBROWN,
                         min=inv.MIN_CBROWN,
                         max=inv.MAX_CBROWN,
                         step=0.05, description='Cbrown (-):', **slide_kwargs)
w_cw = w.FloatSlider(value=inv.MEAN_CW,
                     min=inv.MIN_CW,
                     max=inv.MAX_CW,
                     step=0.001, description='Cw (g/cm²):',
                     readout_format='.3f', **slide_kwargs)
w_cm = w.FloatSlider(value=inv.MEAN_CM,
                     min=inv.MIN_CM,
                     max=inv.MAX_CM,
                     step=0.001, description='Cm (g/cm²):',
                     readout_format='.3f', **slide_kwargs)

w_soil = w.Dropdown(options=soil_types, value=soil_types[0], description='Soil Type')

w_lai = w.FloatSlider(value=inv.MEAN_LAI,
                      min=inv.MIN_LAI,
                      max=inv.MAX_LAI,
                      step=0.1, description='LAI (m²/m²):', **slide_kwargs)
w_hotspot = w.FloatSlider(value=inv.MEAN_HOTSPOT,
                          min=0.001,
                          max=1,
                          step=0.01, description='hotspot (-):', **slide_kwargs)
w_leaf_angle = w.FloatSlider(value=inv.MEAN_LEAF_ANGLE,
                             min=0,
                             max=90,
                             step=1, description='Leaf Angle (deg.):',
                             **slide_kwargs)
w_sza = w.FloatSlider(value=35., min=0, max=89, step=1,
                      description='SZA (deg.):', **slide_kwargs)
w_vza = w.FloatSlider(value=0, min=0, max=89, step=1,
                      description='VZA (deg.):', **slide_kwargs)
w_psi = w.FloatSlider(value=0, min=0, max=180, step=1,
                      description='PSI (deg.):', **slide_kwargs)
w_skyl = w.FloatSlider(value=0.1, min=0, max=1, step=0.01,
                       description='skyl (-):', **slide_kwargs)
w_param = w.Dropdown(options=RANGE_DICT.keys(), value="LAI",
                     description='Variable a evaluar')
w_range = w.FloatRangeSlider(value=RANGE_DICT["LAI"],
                             min=RANGE_DICT["LAI"][0],
                             max=RANGE_DICT["LAI"][1],
                             description="Rango", readout_format='.1f',
                             **slide_kwargs)

srf_list = sorted(glob(os.path.join(SRF_FOLDER, "*.txt")))
sensor_list = [os.path.splitext(os.path.basename(i))[0] for i in srf_list]
w_sensor = w.Dropdown(options=sensor_list, value=sensor_list[0],
                      description='Sensor')

# Widgets for LUT building
w_sims = w.IntSlider(value=5000, min=1000, max=100000, step=100,
                     description="Simulaciones")
w_range_nleaf = w.FloatRangeSlider(value=RANGE_DICT["N_leaf"],
                                 min=RANGE_DICT["N_leaf"][0],
                                 max=RANGE_DICT["N_leaf"][1],
                                 step=0.1,
                                 description="N_leaf",
                                 readout_format='.1f',
                                 **slide_kwargs)
w_range_cab = w.FloatRangeSlider(value=RANGE_DICT["Cab"],
                                 min=RANGE_DICT["Cab"][0],
                                 max=RANGE_DICT["Cab"][1],
                                 step=0.1,
                                 description="Cabr",
                                 readout_format='.1f',
                                 **slide_kwargs)
w_range_car = w.FloatRangeSlider(value=RANGE_DICT["Car"],
                                 min=RANGE_DICT["Car"][0],
                                 max=RANGE_DICT["Car"][1],
                                 step=0.1,
                                 description="Car",
                                 readout_format='.1f',
                                 **slide_kwargs)
w_range_ant = w.FloatRangeSlider(value=RANGE_DICT["Ant"],
                                 min=RANGE_DICT["Ant"][0],
                                 max=RANGE_DICT["Ant"][1],
                                 step=0.1,
                                 description="Ant",
                                 readout_format='.1f',
                                 **slide_kwargs)
w_range_cbrown = w.FloatRangeSlider(value=RANGE_DICT["Cbrown"],
                                 min=RANGE_DICT["Cbrown"][0],
                                 max=RANGE_DICT["Cbrown"][1],
                                 step=0.1,
                                 description="Cbrown",
                                 readout_format='.1f',
                                 **slide_kwargs)
w_range_cbrown = w.FloatRangeSlider(value=RANGE_DICT["Cbrown"],
                                 min=RANGE_DICT["Cbrown"][0],
                                 max=RANGE_DICT["Cbrown"][1],
                                 step=0.1,
                                 description="Cbrown",
                                 readout_format='.1f',
                                 **slide_kwargs)
w_range_cm = w.FloatRangeSlider(value=RANGE_DICT["Cm"],
                                 min=RANGE_DICT["Cm"][0],
                                 max=RANGE_DICT["Cm"][1],
                                 step=0.001,
                                 description="Cm",
                                 readout_format= '.3f',
                                 **slide_kwargs)
w_range_cw = w.FloatRangeSlider(value=RANGE_DICT["Cw"],
                                 min=RANGE_DICT["Cw"][0],
                                 max=RANGE_DICT["Cw"][1],
                                 step=0.001,
                                 description="Cw",
                                 readout_format= '.3f',
                                 **slide_kwargs)
w_range_lai = w.FloatRangeSlider(value=RANGE_DICT["LAI"],
                                 min=RANGE_DICT["LAI"][0],
                                 max=RANGE_DICT["LAI"][1],
                                 step=0.1,
                                 description="LAI",
                                 readout_format= '.1f',
                                 **slide_kwargs)
w_range_leaf_angle = w.FloatRangeSlider(value=RANGE_DICT["leaf_angle"],
                                 min=0,
                                 max=90,
                                 step=1,
                                 description="Leaf Angle",
                                 readout_format= '.0f',
                                 **slide_kwargs)
w_range_hotspot = w.FloatRangeSlider(value=RANGE_DICT["hotspot"],
                                 min=0,
                                 max=1,
                                 step=0.01,
                                 description="hotspot",
                                 readout_format= '.2f',
                                 **slide_kwargs)
w_soils = w.SelectMultiple(options=soil_types,
                           value=soil_types,
                           description='Suelos',
                           rows=20)


def _on_param_change(args):
    var = args["new"]
    if var == "Cm" or var == "Cw":
        w_range.readout_format = '.3f'
        w_range.step = 0.001
    else:
        w_range.readout_format = '.1f'
        w_range.step = 0.1

    if RANGE_DICT[var][1] < w_range.min:
        w_range.min = RANGE_DICT[var][0]
        w_range.max = RANGE_DICT[var][1]
    else:
        w_range.max = RANGE_DICT[var][1]
        w_range.min = RANGE_DICT[var][0]
    w_range.value = RANGE_DICT[var]


w_param.observe(_on_param_change, 'value', type="change")

class ProSailSensitivity(object):

    def __init__(self):
        '''Initialize input variables  with default  values'''
        self.params = PARAM_DICT.copy()
        self.widget_style = {'description_width': "150px"}
        self.widget_layout = widgets.Layout(width='50%')

        # Configure the I/O widgets
        self.w_sliders = {}

        # Load and save configuration buttons
        self.w_sliders["N_leaf"] = widgets.FloatSlider(
            value=inv.MEAN_N_LEAF,
            min=RANGE_DICT["N_leaf"][0],
            max=RANGE_DICT["N_leaf"][1],
            step=0.1,
            description='Structural Parameter (N)',
            style=self.widget_style,
            layout=self.widget_layout)

        self.w_sliders["Cab"] = widgets.FloatSlider(
            value=inv.MEAN_CAB,
            min=RANGE_DICT["Cab"][0],
            max=RANGE_DICT["Cab"][1],
            step=1,
            description='Chlorophyll ($\mu$g/cm²)',
            style=self.widget_style,
            layout=self.widget_layout)

        self.w_sliders["Car"] = widgets.FloatSlider(
            value=inv.MEAN_CAR,
            min=RANGE_DICT["Car"][0],
            max=RANGE_DICT["Car"][1],
            step=1,
            description='Carotenoids ($\mu$g/cm²)',
            style=self.widget_style,
            layout=self.widget_layout)

        self.w_sliders["Ant"] = widgets.FloatSlider(
            value=inv.MEAN_ANT,
            min=RANGE_DICT["Ant"][0],
            max=RANGE_DICT["Ant"][1],
            step=1,
            description='Atocianyns ($\mu$g/cm²)',
            style=self.widget_style,
            layout=self.widget_layout)

        self.w_sliders["Cbrown"] = widgets.FloatSlider(
            value=inv.MEAN_CBROWN,
            min=RANGE_DICT["Cbrown"][0],
            max=RANGE_DICT["Cbrown"][1],
            step=0.01,
            description='Brown Pigments (--)',
            style=self.widget_style,
            layout=self.widget_layout)

        self.w_sliders["Cw"] = widgets.FloatSlider(
            value=inv.MEAN_CW,
            min=RANGE_DICT["Cw"][0],
            max=RANGE_DICT["Cw"][1],
            step=0.001,
            description='Water Conent (g/cm²)',
            style=self.widget_style,
            layout=self.widget_layout)

        self.w_sliders["Cm"] = widgets.FloatSlider(
            value=inv.MEAN_CM,
            min=RANGE_DICT["Cm"][0],
            max=RANGE_DICT["Cm"][1],
            step=0.001,
            description='Dry Matter (g/cm²)',
            style=self.widget_style,
            layout=self.widget_layout)

        self.w_sliders["LAI"] = widgets.FloatSlider(
            value=inv.MEAN_LAI,
            min=RANGE_DICT["LAI"][0],
            max=RANGE_DICT["LAI"][1],
            step=0.25,
            description='Leaf Area Index',
            disabled=True,
            style=self.widget_style,
            layout=self.widget_layout)

        self.w_sliders["leaf_angle"] = widgets.FloatSlider(
            value=inv.MEAN_LEAF_ANGLE,
            min=RANGE_DICT["leaf_angle"][0],
            max=RANGE_DICT["leaf_angle"][1],
            step=1,
            description='Average Leaf Angle (deg)',
            style=self.widget_style,
            layout=self.widget_layout)

        self.w_sliders["hotspot"] = widgets.FloatSlider(
            value=inv.MEAN_HOTSPOT,
            min=RANGE_DICT["hotspot"][0],
            max=RANGE_DICT["hotspot"][1],
            step=0.01,
            description='Leaf Size to Canopy Height shape factor (-)',
            style=self.widget_style,
            layout=self.widget_layout)

        self.w_soil = widgets.Dropdown(
            options=soil_types,
            value=soil_types[0],
            description='Soil Type',
            style=self.widget_style,
            layout=self.widget_layout)

        self.w_sliders["SZA"] = widgets.FloatSlider(
            value=35,
            min=RANGE_DICT["SZA"][0],
            max=RANGE_DICT["SZA"][1],
            step=1,
            description='Sun Zenight Angle (deg.)',
            style=self.widget_style,
            layout=self.widget_layout)


        self.w_sliders["VZA"] = widgets.FloatSlider(
            value=0,
            min=RANGE_DICT["VZA"][0],
            max=RANGE_DICT["VZA"][1],
            step=1,
            description='View Zenith Angle (deg.)',
            style=self.widget_style,
            layout=self.widget_layout)

        self.w_sliders["PSI"] = widgets.FloatSlider(
            value=0,
            min=RANGE_DICT["PSI"][0],
            max=RANGE_DICT["PSI"][1],
            step=1,
            description='Sun-to-view Relative Azimtuh Angle (deg.)',
            style=self.widget_style,
            layout=self.widget_layout)

        self.w_sliders["skyl"] = widgets.FloatSlider(
            value=0.1,
            min=RANGE_DICT["skyl"][0],
            max=RANGE_DICT["skyl"][1],
            step=1,
            description='Fraction of diffuse light (-)',
            style=self.widget_style,
            layout=self.widget_layout)

        self.w_obj_param = widgets.Dropdown(
            options=RANGE_DICT.keys(),
            value="LAI",
        )

        self.w_range = widgets.FloatRangeSlider(
            value=RANGE_DICT["LAI"],
            min=RANGE_DICT["LAI"][0],
            max=RANGE_DICT["LAI"][1],
            description="LAI",
            layout=self.widget_layout)

        self.w_run = widgets.Button(description='Get Spectra')

        # Handle interactions
        self.w_run.on_click(self._update_plot)
        self.w_obj_param.observe(self._on_param_change, 'value', type="change")
        self.w_soil.observe(self._on_soil_change, 'value', type="change")


    def gui(self):

        w_box_leaf = widgets.VBox([widgets.HTML('Leaf Bio-Chemical properties'),
                                   self.w_sliders["N_leaf"],
                                   self.w_sliders["Cab"],
                                   self.w_sliders["Car"],
                                   self.w_sliders["Ant"],
                                   self.w_sliders["Cbrown"],
                                   self.w_sliders["Cm"],
                                   self.w_sliders["Cw"]],
                                  layout=widgets.Layout(width='100%'))

        w_box_canopy = widgets.VBox([widgets.HTML('Canopy Structure'),
                                     self.w_sliders["LAI"],
                                     self.w_sliders["leaf_angle"],
                                     self.w_sliders["hotspot"]])

        w_box_obs = widgets.VBox(
            [widgets.HTML('Illumination and Observation Geometry'),
             self.w_sliders["SZA"],
             self.w_sliders["VZA"],
             self.w_sliders["PSI"],
             self.w_sliders["skyl"]])

        w_box_param = widgets.VBox(
            [widgets.HTML('Parameter to evaluate sensitivity'),
             self.w_obj_param,
             self.w_range])

        # Display widgets
        display(w_box_param, background_color='#EEE')
        display(w_box_leaf, background_color='#EEE')
        display(w_box_canopy, background_color='#EEE')
        display(self.w_soil, background_color='#EEE')
        display(w_box_obs, background_color='#EEE')
        display(self.w_run)


    def get_canopy_spectra(self):
        obj_param = self.w_obj_param.value
        # Replace the given value
        for key, val in self.params.items():
            if key == obj_param:
                self.params[key] = np.linspace(*self.w_range.value, N_STEPS)
            elif key == "soil":
                pass
            else:
                self.params[key] = np.full(N_STEPS, self.w_sliders[key].value)

        wls, rho_leaf, tau_leaf = pro.prospectd_vec(self.params["N_leaf"],
                                                    self.params["Cab"],
                                                    self.params["Car"],
                                                    self.params["Cbrown"],
                                                    self.params["Cw"],
                                                    self.params["Cm"],
                                                    self.params["Ant"])

        lidf = sail.calc_lidf_campbell_vec(self.params["leaf_angle"])
        rsoil = np.genfromtxt(os.path.join(SOIL_FOLDER, self.params["soil"]))
        # wl_soil=rsoil[:,0]
        rsoil_vec = np.tile(np.array(rsoil[:, 1]), (N_STEPS, 1))

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
         _] = sail.foursail_vec(self.params["LAI"],
                                self.params["hotspot"],
                                lidf,
                                self.params["SZA"],
                                self.params["VZA"],
                                self.params["PSI"],
                                rho_leaf.T,
                                tau_leaf.T,
                                rsoil_vec.T)
        r2 = rdot * self.params["skyl"] + rsot * (1 - self.params["skyl"])
        r2 = r2.T
        bare = self.params["LAI"] == 0
        r2[bare, :] = rsoil[:, 1]

        return wls, r2


    def _on_param_change(self, args):

        self.w_sliders[args["new"]].disabled = True
        self.w_sliders[args["old"]].disabled = False
        self.w_range.description = args["new"]
        if RANGE_DICT[args["new"]][0] >= self.w_range.max:
            self.w_range.max = RANGE_DICT[args["new"]][1]
            self.w_range.min = RANGE_DICT[args["new"]][0]
        else:
            self.w_range.min = RANGE_DICT[args["new"]][0]
            self.w_range.max = RANGE_DICT[args["new"]][1]
        self.w_range.value = RANGE_DICT[args["new"]]

    def _on_soil_change(self, args):
        soil_file = "%s.txt"%args["new"]
        self.params["soil"] = soil_file


    def _update_plot(self, b):
        obj_param = self.w_obj_param.value
        wls, r2 = self.get_canopy_spectra()
        plot_sensitivity(wls, r2, obj_param, self.params[obj_param], taus=None)


def update_prospect_spectrum(N_leaf, Cab, Car, Ant, Cbrown, Cw, Cm):
    wls, rho, tau = pro.prospectd(N_leaf, Cab, Car, Cbrown, Cw, Cm, Ant)
    plot_spectrum(wls, rho, tau=tau)
    return wls, rho, tau


def update_soil_spectrum(soil_name):
    soil_file = "%s.txt"%soil_name
    rsoil = np.genfromtxt(os.path.join(SOIL_FOLDER, soil_file))
    plot_spectrum(rsoil[:, 0], rsoil[:, 1])
    return rsoil[:, 0], rsoil[:, 1]


def update_prosail_spectrum(N_leaf, Cab, Car, Ant, Cbrown, Cw, Cm,
                            lai, hotspot, leaf_angle,
                            sza, vza, psi, skyl,
                            soil_name):
    # Replace the given value
    soil_file = "%s.txt"%soil_name
    wls, rho_leaf, tau_leaf = pro.prospectd(N_leaf, Cab, Car, Cbrown,
                                            Cw, Cm, Ant)
    rsoil = np.genfromtxt(os.path.join(SOIL_FOLDER, soil_file))
    rsoil = rsoil[:, 1]
    lidf = sail.calc_lidf_campbell_vec(leaf_angle)

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
     _] = sail.foursail(lai, hotspot, lidf, sza, vza, psi, rho_leaf, tau_leaf,
                        rsoil)
    r2 = rdot * skyl + rsot * (1 - skyl)
    plot_spectrum(wls, r2)
    return wls, r2


def update_4sail_spectrum(lai, hotspot, leaf_angle, sza, vza, psi, skyl,
                          leaf_spectrum, soil_spectrum):

    wls, rho_leaf, tau_leaf = leaf_spectrum.result
    rho_soil = soil_spectrum.result[1]
    lidf = sail.calc_lidf_campbell_vec(leaf_angle)

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
     _] = sail.foursail(lai, hotspot, lidf, sza, vza, psi, rho_leaf, tau_leaf,
                        rho_soil)
    r2 = rdot * skyl + rsot * (1 - skyl)
    plot_spectrum(wls, r2)
    return wls, r2


def plot_spectrum(wls, rho, tau=None):

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(wls, rho, "k")
    if tau is not None:
        secax = ax.secondary_yaxis('right', functions=(lambda x: 1 - x,
                                                       lambda x: 1 - x),
                                   )
        secax.set_ylabel('Transmittance', color="blue")
        secax.tick_params(axis='y', colors='blue')
        ax.plot(wls, 1. - tau, "b")
    ax.set_ylabel('Reflectance')
    ax.set_ylim((0, 1))
    ax.set_xlim((400, 2500))
    ax.set_xlabel('Wavelength (nm)')
    for region, wls in REGIONS.items():
        ax.axvline(x=wls[1], c="silver",  ls="--")
        ax.text(np.mean(wls), 0.95, region, size="large", ha="center")
    plt.tight_layout()
    plt.show()


def plot_sensitivity(wls, rhos, param_name, param_values, taus=None):
    plt.figure(figsize=FIGSIZE)
    colors = plt.cm.RdYlGn(np.linspace(0, 1, param_values.shape[0]))
    for i, value in enumerate(param_values):
        plt.plot(wls, rhos[i], color=colors[i], label=np.round(value, 3))
        if taus is not None:
            plt.plot(wls, 1. - taus[i], color=colors[i])

    plt.legend(title=param_name)
    plt.ylabel('Reflectance')
    plt.ylim((0, 1))
    plt.xlabel('Wavelength (nm)')
    plt.xlim((400, 2500))
    for region, wls in REGIONS.items():
        plt.axvline(x=wls[1], c="silver",  ls="--")
        plt.text(np.mean(wls), 0.95, region, size="large", ha="center")
    plt.tight_layout()
    plt.show()


def prosail_sensitivity(N_leaf, Cab, Car, Ant, Cbrown, Cw, Cm,
                        lai, hotspot, leaf_angle, sza, vza, psi, skyl,
                        soil_name, var, value_range):
    params = {"N_leaf": N_leaf, "Cab": Cab, "Car": Car, "Cbrown": Cbrown,
              "Cw": Cw, "Cm": Cm, "Ant": Ant,
              "leaf_angle": leaf_angle, "LAI": lai, "hotspot": hotspot,
              "SZA": sza, "VZA": vza, "PSI": psi, "skyl": skyl}

    # Replace the given value
    for key, val in params.items():
        if key == var:
            params[key] = np.linspace(*value_range, N_STEPS)
        elif key == "soil":
            pass
        else:
            params[key] = np.full(N_STEPS, params[key])

    wls, rho_leaf, tau_leaf = pro.prospectd_vec(params["N_leaf"],
                                                params["Cab"],
                                                params["Car"],
                                                params["Cbrown"],
                                                params["Cw"],
                                                params["Cm"],
                                                params["Ant"])

    lidf = sail.calc_lidf_campbell_vec(params["leaf_angle"])
    soil_file = "%s.txt"%soil_name
    rsoil = np.genfromtxt(os.path.join(SOIL_FOLDER, soil_file))
    # wl_soil=rsoil[:,0]
    rsoil_vec = np.tile(np.array(rsoil[:, 1]), (N_STEPS, 1))

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
     _] = sail.foursail_vec(params["LAI"],
                            params["hotspot"],
                            lidf,
                            params["SZA"],
                            params["VZA"],
                            params["PSI"],
                            rho_leaf.T,
                            tau_leaf.T,
                            rsoil_vec.T)
    r2 = rdot * params["skyl"] + rsot * (1 - params["skyl"])
    r2 = r2.T
    bare = params["LAI"] == 0
    r2[bare, :] = rsoil[:, 1]
    plot_sensitivity(wls, r2, var, params[var])
    return wls, r2

def sensor_sensitivity(sensor, spectra):

    srf_file = os.path.join(SRF_FOLDER, sensor + ".txt")
    srfs = np.genfromtxt(srf_file, dtype=None, names=True)
    srf = []
    wls = srfs["SR_WL"]
    rho_full = spectra.result[1]
    rho_sensor = []
    wls_sensor = []
    plt.figure(figsize=FIGSIZE)
    for band in srfs.dtype.names[1:]:
        wls_sensor.append(np.sum(wls * srfs[band]) / np.sum(srfs[band]))
        rho_sensor.append(np.sum(rho_full * srfs[band], axis=1) / np.sum(srfs[band]))
        valid = srfs[band] > 0
        plt.plot(wls[valid], srfs[band][valid], label=band)

    plt.legend(title="%s bands"%sensor, loc="upper right")
    plt.ylabel('Spectral Response')
    plt.ylim((0, 1))
    plt.xlabel('Wavelength (nm)')
    for region, wls in REGIONS.items():
        plt.axvline(x=wls[1], c="silver",  ls="--")
        plt.text(np.mean(wls), 0.95, region, size="large", ha="center")
    plt.xlim((400, 2501))
    plt.tight_layout()
    rho_sensor = np.asarray(rho_sensor).T
    wls_sensor = np.asarray(wls_sensor)
    val_range = np.linspace(*spectra.children[-2].value, N_STEPS)
    plot_sensitivity(wls_sensor, rho_sensor, spectra.children[-3].value, val_range)

def build_random_simulations(n_sim, n_leaf_range, cab_range, car_range, ant_range,
                             cbrown_range, cw_range, cm_range,
                             lai_range, hotspot_range, leaf_angle_range,
                             sza, vza, psi, skyl,
                             soil_names, sensor):

    param_bounds = {"N_leaf": n_leaf_range, "Cab": cab_range, "Car": car_range,
                    "Ant": ant_range, "Cbrown": cbrown_range, "Cw": cw_range,
                    "Cm": cm_range, "LAI": lai_range, "leaf_angle": leaf_angle_range,
                    "hotspot": hotspot_range}

    distribution = inv.SALTELLI_DIST

    params_orig = inv.build_prosail_database(n_sim,
                                             param_bounds=param_bounds,
                                             distribution=distribution)

    soil_files = [os.path.join(SOIL_FOLDER, '%s.txt'%i) for i in soil_names]
    n_soils = len(soil_files)
    soil_spectrum = []
    for soil_file in soil_files:
        r = np.genfromtxt(soil_file)
        soil_spectrum.append(r[:, 1])

    n_simulations = params_orig["LAI"].size
    multiplier = int(np.ceil(float(n_simulations / n_soils)))
    soil_spectrum = np.asarray(soil_spectrum * multiplier)
    soil_spectrum = soil_spectrum[:n_simulations]
    soil_spectrum = soil_spectrum.T

    srf_file = os.path.join(SRF_FOLDER, sensor + ".txt")
    srfs = np.genfromtxt(srf_file, dtype=None, names=True)
    srf = []
    band_names = []
    wls_sensor = []
    wls = np.arange(400, 2501)
    for band in srfs.dtype.names[1:]:
        srf.append(srfs[band])
        band_names.append(band)
        wls_sensor.append(np.sum(wls * srfs[band]) / np.sum(srfs[band]))

    print('Building ProspectD+4SAIL database... This could take some time')
    rho_canopy_vec, params = inv.simulate_prosail_lut(params_orig,
                                                      wls,
                                                      soil_spectrum,
                                                      srf=srf,
                                                      skyl=skyl,
                                                      sza=np.full(n_simulations, sza),
                                                      vza=np.full(n_simulations, vza),
                                                      psi=np.full(n_simulations, psi),
                                                      calc_FAPAR=True,
                                                      reduce_4sail=True)

    rho_canopy_vec = pd.DataFrame(rho_canopy_vec, columns=band_names)
    params = pd.DataFrame(params)
    result = pd.concat([params, rho_canopy_vec], axis=1)
    del params, rho_canopy_vec
    out_file = os.path.join(OUTPUT_FOLDER, "prosail_simulations.csv")
    if not os.path.isdir(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    result.to_csv(out_file)
    print('Simulations saved in %s'%out_file)
    print("Plotting VI's relationships")
    red = find_band_pos(band_names, wls_sensor, 650)
    nir = find_band_pos(band_names, wls_sensor, 850)
    rededge = find_band_pos(band_names, wls_sensor, 715)
    swir = find_band_pos(band_names, wls_sensor, 1600)

    ndvi = (result[nir] - result[red]) / (result[nir] + result[red])
    ndre = (result[nir] - result[rededge]) / (result[nir] + result[rededge])
    ndwi = (result[nir] - result[swir]) / (result[nir] + result[swir])

    fig, axs = plt.subplots(ncols=4, figsize=FIGSIZE, sharey=True)
    axs[0].scatter(result["LAI"], ndvi, c="blue", s=3, alpha=0.5)
    axs[0].set_xlabel('LAI')
    axs[0].set_title('NDVI')
    axs[1].scatter(result["fAPAR"], ndvi, c="blue", s=3, alpha=0.5)
    axs[1].set_xlabel('fAPAR')
    axs[1].set_title('NDVI')
    axs[2].scatter(result["Cab"], ndre, c="blue", s=3, alpha=0.5)
    axs[2].set_xlabel('C$_{a+b}$ ($\mu$g/cm²)')
    axs[2].set_title('NDRE')
    axs[3].scatter(result["Cw"], ndwi, c="blue", s=3, alpha=0.5)
    axs[3].set_xlabel('C$_{w}$ (g/cm²)')
    axs[3].set_title('NDWI')
    axs[0].set_ylabel('Índice de Vegetación')
    axs[0].set_ylim((0, 1))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.show()


def find_band_pos(band_names, wls_sensor, wl):
    diff = np.abs(np.asarray(wls_sensor) - wl)
    min_diff = np.min(diff)
    pos = int(np.where(diff == min_diff)[0])
    return band_names[pos]

