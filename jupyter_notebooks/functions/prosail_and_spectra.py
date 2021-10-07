import matplotlib.pyplot as plt
from pypro4sail import pypro4sail as fs
from pypro4sail import machine_learning_regression as inv
from pypro4sail import four_sail as sail
from pypro4sail import prospect as pro
import numpy as np
import os
import ipywidgets as widgets
from IPython.display import display, clear_output
print("Gracias! librerías correctamente importadas")
print("Puedes continuar con las siguientes tareas")

# Generate the list with VZAs (from 0 to 89)
VZAS = np.arange(0, 99)
SOIL_FOLDER = os.path.join(os.path.dirname(fs.__file__),
                           "spectra", "soil_spectral_library")

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
              "PSI": (0, 90),
              "VZA": (0, 89),
              "skyl": (0, 1)}

N_STEPS = 10

SOIL_TYPES = ["alfisol.fragiboralf.coarse.86P1994",
               "alfisol.haploxeralf.coarse.87P313",
               "alfisol.haplustalf.coarse.87P3671",
               "alfisol.paleustalf.coarse.87P2410",
               "aridisol.calciorthid.coarse.79P1536",
               "aridisol.camborthid.coarse.87P337",
               "aridisol.gypsiorthid.coarse.82P2695",
               "aridisol.haplargid.coarse.89P1793",
               "aridisol.salorthid.coarse.79P1530",
               "aridisol.torripsamment.coarse.90P0142",
               "entisol.quartzipsamment.coarse.87P706",
               "entisol.torripsamment.coarse.0015",
               "entisol.ustifluvent.coarse.82P2230",
               "inceptisol.cryumbrept.coarse.87P3855",
               "inceptisol.dystrochrept.coarse.88P2535",
               "inceptisol.haplumbrept.coarse.86P4561",
               "inceptisol.plaggept.coarse.85P3707",
               "inceptisol.xerumbrept.coarse.87P325",
               "mollisol.agialboll.coarse.85P5339",
               "mollisol.agriudoll.coarse.87P757",
               "mollisol.argiustoll.coarse.90P128s",
               "mollisol.cryoboroll.coarse.85P4663",
               "mollisol.haplaquoll.coarse.86P4603",
               "mollisol.hapludoll.coarse.87P764",
               "mollisol.haplustall.coarse.85P4569",
               "mollisol.paleustoll.coarse.90P186s",
               "spodosol.cryohumod.coarse.87P4264",
               "utisol.hapludult.coarse.87P707",
               "vertisol.chromoxerert.coarse.88P475"]

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
            options=SOIL_TYPES,
            value=SOIL_TYPES[0],
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
        soil_file = "jhu.becknic.soil.%s.spectrum.txt"%args["new"]
        self.params["soil"] = soil_file


    def _update_plot(self, b):
        obj_param = self.w_obj_param.value
        wls, r2 = self.get_canopy_spectra()
        plot_sensitivity(wls, r2, obj_param, self.params[obj_param], taus=None)
        # colors = plt.cm.RdYlGn(np.linspace(0, 1, self.params[obj_param].shape[0]))
        # fig = plt.figure(figsize=(12.0, 6.0))
        # ax = fig.add_subplot(1, 1, 1)
        # for i, value in enumerate(self.params[obj_param]):
        #     ax.plot(wls, r2[i], color=colors[i], label=np.round(value, 3))
        #
        # ax.legend(title=obj_param)
        # ax.set_ylabel('Reflectance')
        # ax.set_xlabel('Wavelenght (nm)')
        # ax.set_xlim((np.min(wls), np.max(wls)))
        # ax.set_ylim((0, 1))


def update_prospect_spectrum(N_leaf, Cab, Car, Ant, Cbrown, Cw, Cm):
    wls, rho, tau = pro.prospectd(N_leaf, Cab, Car, Cbrown, Cw, Cm, Ant)
    plot_spectrum(wls, rho, tau=tau)
    return wls, rho, tau


def update_soil_spectrum(soil_name):
    soil_file = "jhu.becknic.soil.%s.spectrum.txt"%soil_name
    rsoil = np.genfromtxt(os.path.join(SOIL_FOLDER, soil_file))
    plot_spectrum(rsoil[:, 0], rsoil[:, 1])
    return rsoil[:, 0], rsoil[:, 1]


def update_prosail_spectrum(N_leaf, Cab, Car, Ant, Cbrown, Cw, Cm,
                            lai, hotspot, leaf_angle,
                            sza, vza, psi, skyl,
                            soil_name):
    # Replace the given value
    soil_file = "jhu.becknic.soil.%s.spectrum.txt"%soil_name
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
                          rho_leaf, tau_leaf, rho_soil,
                          wls=np.arange(400, 2501)):
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
    plt.figure(figsize = (12.0, 6.0))
    plt.plot(wls, rho, "k")
    if tau is not None:
        plt.plot(wls, 1. - tau, "b")
    plt.ylabel('Reflectance')
    plt.ylim((0, 1))
    plt.xlim((np.min(wls), np.max(wls)))
    plt.xlabel('Wavelength (nm)')
    plt.tight_layout()
    # plt.show()


def plot_sensitivity(wls, rhos, param_name, param_values, taus=None):
    plt.figure(figsize = (12.0, 6.0))
    colors = plt.cm.RdYlGn(np.linspace(0, 1, param_values.shape[0]))
    for i, value in enumerate(param_values):
        plt.plot(wls, rhos[i], color=colors[i], label=np.round(value, 3))
        if taus is not None:
            plt.plot(wls, 1. - taus[i], color=colors[i])

    plt.legend(title=param_name)
    plt.ylabel('Reflectance')
    plt.ylim((0, 1))
    plt.xlabel('Wavelength (nm)')
    plt.xlim((np.min(wls), np.max(wls)))
    plt.tight_layout()
    plt.show()

