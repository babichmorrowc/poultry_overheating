# Exposure: set to 1 everywhere
# Hazard: UKCP18 temperature
# Vulnerability: 
# Risk: annual expected days where the indoor temperature exceeds __

import warnings
# import sys
# import os
import glob
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs

#import climada functions
from climada.hazard import Hazard
from climada.entity import Exposures
from climada.entity import ImpactFunc, ImpactFuncSet
from climada.entity import Entity
from climada.engine import Impact, ImpactCalc
# from climada.entity import Measure, MeasureSet
# from climada.engine import CostBenefit
# from climada.engine.cost_benefit import risk_aai_agg
# from climada.entity import DiscRates

# Set file paths
DATA_DIR = 'data/'
OUT_DIR = 'output/'

# Function to plot a location on the map given its index (for troubleshooting)
def plot_index(file, index):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    # File with lat/lon points
    dat = pd.read_csv(file)
    # Plot all points in grey
    ax.scatter(dat['longitude'], dat['latitude'], c="grey", s=12)
    # Subset to desired index
    dat_ind = dat.loc[index]
    ax.scatter(dat_ind['longitude'],dat_ind['latitude'],c="red",s=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.coastlines()
    return fig

def read_hazard(warming_level, ens_mem):
    """
    Read in hazard data

    Inputs
    ------
       warming_level: current, WL2, WL4
       ens_mem: one of 12 UKCP ensemble members

    Returns
    -------
         netcdf_file_path: path of data used
         netcdf_file: Dataset containing hazard data
    """
    if warming_level == 'current':
        netcdf_file_path = glob.glob(
                DATA_DIR + f'/UKCP_BC/Timeseries_{ens_mem}_tasmax_1998*')
    else:
        netcdf_file_path = glob.glob(
            DATA_DIR + f'/UKCP_BC/Timeseries_{ens_mem}_tasmax*{warming_level}*')

    print(netcdf_file_path)

    # load in the hazard data (mean_temperature)

    netcdf_file = Dataset(netcdf_file_path[0])
    return netcdf_file_path[0], netcdf_file

def define_hazard(file_name, nc1, variable, haz_type, custom_nyears=False):
    """
    Define the hazard data and read it in from netcdf files to create a Heat
    stress  instance of a hazard object

    Inputs
    ------
        file_name: (string) name of netcdf file to be read in
        nc1: Dataset of hazard data
        variable: variable of interest in the netcdf file
        haz_type: user defined hazard category
        custom_nyears: set to True if want to alter number of years, e.g. to
            remove weekends or holidays

    Returns
    -------
         hazard: (Hazard Object)
    """
    nyears = round(nc1.dimensions['time'].size)/ 360 #climate day with 360 days per year
    if custom_nyears:
        nyears = (360*nyears)/CUSTOM_NYEARS
        print(f'using custom number of years: {nyears}')

    # Variables that help defined the Heat stress instance
    hazard_args = {'intensity_var': variable,
                   'event_id': np.arange(1, len(nc1.variables['time'])+1),
                   'frequency': np.full(len(nc1.variables['time']), 1/nyears),
                   'haz_type': haz_type,
                   'description': 'Hazard data',
                   'replace_value': np.nan,
                   'fraction_value': 1.0}

    # read in hazard data from netcdf and use the previously defined variables
    # to help define the data
    hazard1 = Hazard.from_netcdf(file_name, **hazard_args)

    hazard1.check()  #This needs to come before the plots

    # NEED TO FIX LATER
    # Cast event name to string if necessary
    # print('Event ids:')
    # print(haz.event_id)
    # print('Event name:')
    # print(haz.event_name)
    num_ev = hazard1.event_id.size
    event_names = hazard1.event_name[:num_ev]
    if not pd.api.types.is_string_dtype(event_names):
        warnings.warn("Some event names are not str will be converted to str", UserWarning)
        event_names = list(map(str, event_names))
    hazard1.event_name = event_names
    print(hazard1.event_name)

    return hazard1

def read_exposures_csv(exposure_csv):
    """
        read in exposure information from a csv file
        (e.g. location of primary or distribution substations)

        Inputs
        ------
           data: csv with lon and lat

        Returns
        -------
             exposure information in climada format
        """

    assets = pd.read_csv(exposure_csv)

    # drop missing lines
    assets= assets.dropna(how="all")

    # set the value of the asset to be 1. This represents 1 day of over heating.
    # assets["value"] = 1

    # subset the data we want and set appropriate variable names
    # data = assets[['latitude','longitude', 'value']]#
    data = assets

    # create exposure class
    exp = Exposures(data)
    return exp

def exposure_instance(exp, exp_unit):
    """
    Create an exposure instance of the Exposures class for Heat stress and
    produce some plots

    Inputs
    ------
       exposure instance created using CLIMADAs Exposures function


    Returns
    -------
         exp: (Object) exposure instance for an exposure class
    """

    # always apply the check() method in the end. It puts in metadata that has
    # not been assigned, and causes an error if missing mandatory data
    exp.check()

    #Set the value unit
    exp.value_unit = exp_unit

    return exp

def set_entity(exp_instance):
    """
    Put the exposures into the Entity class

    Inputs
    ------
       exp_instance: (Object) exposure instance for an exposure class

    Returns
    -------
         ent: (Object)  an Entity class with an instance of exposures
    """
    ent = Entity()
    ent.exposures = exp_instance
    return ent

def calc_impact(exp1, impfset1, hazard1):
    """
    Create an impact class

    Inputs
    ------
       exp1: (Object)  an Exposures class with a Heat stress instance of
         exposures
       impfset1: (Object)  an ImpactFuncSet class with impact function instance for Heat stress
       hazard1: (Object)  a Hazard class

    Returns
    -------
         imp1: (Object) an Object that contains the results of impact
         calculations
    """

    impcalc1 = ImpactCalc(exp1, impfset1, hazard1)
    imp1 = impcalc1.impact(save_mat = 'True')

    print(np.shape(imp1.imp_mat.toarray()))
    return imp1

def calc_impact_ent(ent1, hazard1):
    """
    Create an impact class

    Inputs
    ------
       ent1: (Object)  an Entity class with a Heat stress instance of
         exposures and impact function instance for Heat stress

    Returns
    -------
         imp1: (Object) an Object that contains the results of impact
         calculations
    """

    imp1 = Impact()

    imp1.calc(ent1.exposures, ent1.impact_funcs, hazard1, save_mat='True')
    print(np.shape(imp1.imp_mat.toarray()))
    return imp1


##########################################################################################################################################
# Start with one ensemble member
ens_mem = '01'

# Define hazard parameters
variable = 'tasmax'
haz_type = 'max_temp'

# Define exposure parameters
exp_unit = 'days'

# Define vulnerability parameters
int_unit = 'degC'

# Define temperature threshold
threshold_temp = 25
# Temperature differential between outdoor and indoor
# temp_diff = 3

# Read in the exposure data
# set exposure to 1 in every cell
# representing possibility of a poultry farm in every location
exposure_path = DATA_DIR + 'exposure_1.csv'
exp = read_exposures_csv(exposure_path)
exp_inst = exposure_instance(exp, exp_unit)

# Plot exposure
# This is hideous but at least it shows there is one dot in every cell
exp.plot_scatter()
plt.title('Exposure - Poultry Farms')
plt.show()

for warming_level in ['current', 'WL2', 'WL4']:
    # read in hazard data
    nc_path, nc = read_hazard(warming_level=warming_level, ens_mem=ens_mem)
    # Get minimum and maximum temperature
    min_temp = np.nanmin(nc.variables[variable][:])
    max_temp = np.nanmax(nc.variables[variable][:])

    # define hazard
    hazard = define_hazard(nc_path,nc,variable,haz_type)

    # Plot hazard
    # hazard.plot_intensity(event = -1) # intensity of the largest event
    # hazard.plot_intensity(event = 1000) # intensity of event 1000
    hazard.plot_intensity(event = 0, vmin=22, vmax=45) # greatest intensity for each point
    plt.savefig(
          OUT_DIR + 'figures/hazard_ens_' + ens_mem.zfill(2) + "_" + str(warming_level) + '.png')
    plt.show(block=False)
    plt.close()

    for temp_diff in [3, 4, 5]:
        # add your impact function to the impact function set
        impf_set = ImpactFuncSet()
        # Create a step function for the impact function
        imp_fun = ImpactFunc().from_step_impf(intensity = (min_temp, threshold_temp - temp_diff, max_temp),
                                              haz_type = haz_type)
        impf_set.append(imp_fun)

        # calculate impact
        imp = calc_impact(exp, impf_set, hazard)
        # Has dimensions of (# of years * 360 days in year) x (number of exposure points)
        # Save impact
        imp.write_csv(OUT_DIR + "/ens_" + ens_mem.zfill(2) +
                            "_" + str(warming_level) + "_tempdiff" + str(temp_diff) + ".csv")

        # Plot the impact function
        # Check if impact function plot exists already
        imp_fun_plot_file = OUT_DIR + 'figures/impf_ens_tempdiff' + str(temp_diff) + '.png'
        if not glob.glob(imp_fun_plot_file):
            impf_set.plot()
            plt.xlabel('Max Indoor Temperature ($^\circ$C)') # fix this
            plt.title('Vulnerability Function - Poultry Heat Stress')
            plt.savefig(imp_fun_plot_file, dpi=500)
            plt.show(block=False)
            plt.close()

        # Impact
        imp.plot_basemap_eai_exposure(vmin = 0, vmax = 165, pop_name = False, buffer = 50000)
        # setting buffer to zoom out is super slow, would love to fix this in CLIMADA later
        map_ax = plt.gcf().get_axes()[0] # first axis is the map, second is the colorbar
        map_ax.set_title('')
        plt.title('')
        plt.savefig(
              OUT_DIR + 'figures/impact_ens_' + ens_mem.zfill(2) + "_" + str(warming_level) + "_tempdiff" + str(temp_diff) + '.png')
        plt.show(block=False)
        plt.close()

# Want to compare EAI in a single location
# plot_index(DATA_DIR + 'exposure_1.csv', 545)
# plt.show()
east_anglia_loc = 545
# Read in impact data
# 2 degree warming, 3 degree temperature differential:
imp_2degwarm_tempdiff3 = Impact.from_csv(OUT_DIR + 'ens_01_WL2_tempdiff3.csv')
# 2 degree warming, 4 degree temperature differential:
imp_2degwarm_tempdiff4 = Impact.from_csv(OUT_DIR + 'ens_01_WL2_tempdiff4.csv')
# 2 degree warming, 5 degree temperature differential:
imp_2degwarm_tempdiff5 = Impact.from_csv(OUT_DIR + 'ens_01_WL2_tempdiff5.csv')

imp_2degwarm_tempdiff3.eai_exp[east_anglia_loc] # 82.2
imp_2degwarm_tempdiff4.eai_exp[east_anglia_loc] # 99.90
imp_2degwarm_tempdiff5.eai_exp[east_anglia_loc] # 116.85