# Exposure: set to 1 everywhere
# Hazard: UKCP18 temperature
# Vulnerability: 
# Risk: annual expected days where the indoor temperature exceeds __

import warnings
import sys
import os
import glob
from netCDF4 import Dataset
import numpy as np
import pandas as pd

#import climada functions
from climada.hazard import Hazard
from climada.entity import Exposures
from climada.entity import ImpactFunc, ImpactFuncSet
from climada.entity import Entity
from climada.engine import Impact, ImpactCalc
from climada.entity import Measure, MeasureSet
from climada.engine import CostBenefit
from climada.engine.cost_benefit import risk_aai_agg
from climada.entity import DiscRates

# Set file paths
DATA_DIR = 'data/'

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
    assets["value"] = 1

    # subset the data we want and set appropriate variable names
    data = assets[['latitude','longitude', 'value']]

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

##########################################################################################################################################
# Start with one ensemble member
ens_mem = '01'

# Define hazard parameters
# warming_level = 'WL4'
variable = 'tasmax'
haz_type = 'Max Temperature'

# Define exposure parameters
exp_unit = 'days'

# Define vulnerability parameters
int_unit = 'degC'

# read in hazard data
nc = Dataset(DATA_DIR + f'{ens_mem}/tasmax_rcp85_land-cpm_uk_5km_{ens_mem}_day_20301201-20401130.nc')
# Get minimum and maximum temperature
min_temp = np.nanmin(nc.variables[variable][:])
max_temp = np.nanmax(nc.variables[variable][:])
# Threshold temperature for poultry heat stress
threshold_temp = 25
# Temperature differential between outdoor and indoor
temp_diff = 3

# define hazard
hazard = define_hazard(DATA_DIR + f'{ens_mem}/tasmax_rcp85_land-cpm_uk_5km_{ens_mem}_day_20301201-20401130.nc',
                       nc, variable,haz_type)

# set exposure to 1 in every cell
# representing possibility of a poultry farm in every location
exposure_path = DATA_DIR + 'exposure_1.csv'
exp = read_exposures_csv(exposure_path)
exp_inst = exposure_instance(exp, exp_unit)

# add your impact function to the impact function set
impf_set = ImpactFuncSet()
# Create a step function for the impact function
imp_fun = ImpactFunc().from_step_impf(intensity = (min_temp, threshold_temp + temp_diff, max_temp), haz_type = haz_type)
impf_set.append(imp_fun)

# calculate impact
imp = calc_impact(exp, impf_set, hazard)
# I'm getting this error: AttributeError: Impact calculation not possible. No impact functions found for hazard type Max Temperature in exposures.