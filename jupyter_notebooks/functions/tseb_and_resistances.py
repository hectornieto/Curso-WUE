import matplotlib.pyplot as plt
from pyTSEB import TSEB
from pyTSEB import MO_similarity as mo
from pyTSEB import wind_profile as wind
from pyTSEB import resistances as res
import numpy as np

# Generate the list with VZAs (from 0 to 89)
VZAS = np.arange(0, 90)

def plot_fveg(lai, chi):
    plt.rcParams['figure.figsize'] = (7.0, 5.0)
    # Create an empty list with the returning f_c for each canopy
    fc = TSEB.calc_F_theta_campbell(VZAS, lai, x_LAD=chi)
    fc_sph = TSEB.calc_F_theta_campbell(VZAS, lai, x_LAD=1)
    # Plot the results
    plt.plot(VZAS, fc, 'r')
    plt.plot(VZAS, fc_sph, 'k--', label='Spherical')
    plt.xlabel('VZA (degrees)')
    plt.ylabel('$f_c$')
    plt.ylim((0, 1.05))
    plt.legend(loc='lower right')


U = 1.  # set wind speed (measured at 10m above the canopy)
Z_U = 2.
H_C = 1.  # Canopy height
LEAF_WIDTH = 0.1  # Leaf width (m)
# Create list of heights
ZS = np.linspace(0, Z_U, 100)

def plot_profile(lai, l_mo):
    def wind_profile(lai, l_mo=np.float('inf')):
        # Estimate surface roughness
        z_0m, d_0 = res.calc_roughness(lai, H_C)
        # Calculate the friction velocity
        u_friction = mo.calc_u_star(U, Z_U, l_mo, d_0, z_0m)
        upper = ZS >= H_C
        u_z = np.ones(ZS.shape)
        u_c = wind.calc_u_C_star(u_friction, H_C, d_0, z_0m, l_mo)
        u_z[upper] = wind.calc_u_C_star(u_friction, ZS[upper], d_0, z_0m, l_mo)
        u_z[~upper] = wind.calc_u_Goudriaan(u_c, H_C, lai, LEAF_WIDTH, ZS[~upper])

        return u_z, u_c

    # Plots the wind profile for given stability lenght compared to a neutral atmosphere
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    # Estimate the wind profile for given LAI and MO Length
    u_z, u_c, = wind_profile(lai, l_mo)
    plt.plot(u_z, ZS, 'b')
    # Plot the ufriction wind canopy
    plt.plot(u_c, H_C, 'b*', label='$U_c$')
    # Estimate the wind profile for given LAI neutral stability
    u_z, u_c, = wind_profile(lai, l_mo=float('inf'))
    plt.plot(u_z, ZS, 'r', label='neutral')
    # Plot the ufriction wind canopy
    plt.plot(u_c, H_C, 'r*', label='Uc neutral')
    # Plot the canopy windspeed according to the two different methods
    plt.legend(loc='upper left')
    plt.xlim((0, U))
    plt.ylim((0, Z_U))
    plt.xlabel('Wind Speed (m)')
    plt.ylabel('$z/h_c$ (m)')


US = np.linspace(0.50, 5, 100)
def plot_resistances(lai, hc, l_mo, leaf_width, z0_soil, delta_t):
    def calc_resistances(lai, hc, l_mo, leaf_width, z0_soil, delta_t,
                        resistance_flag=0):
        rs_list = []
        rx_list = []
        ra_list = []
        z_0m, d_0 = res.calc_roughness(lai, hc)
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
    plt.rcParams['figure.figsize'] = (9.0, 9.0)
    fig, axs = plt.subplots(3, 1, sharex=True)
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