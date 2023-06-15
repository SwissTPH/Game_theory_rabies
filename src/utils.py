# Imports
import numpy as np
import pandas as pd
import networkx as nx
from typing import List
import os

# Creation of results folders
outdirs = ['../jobs']
for outdir in outdirs:
    if not os.path.exists(outdir):
        os.mkdir(outdir)


##################### Variables #########################

# List of countries ISO3 codes
countries_codes_list = ['DZA', 'AGO', 'BEN', 'BWA', 'BFA', 'BDI', 'CMR', 'CAF', 'TCD', 'COD', 'COG', 'CIV', 'DJI', 'EGY',
             'GNQ','ERI','SWZ','ETH','GAB','GMB','GHA','GIN','GNB','KEN','LSO','LBR','LBY','MWI',
             'MLI','MRT','MAR','MOZ','NAM','NER','NGA','RWA','SEN','SLE','SOM','ZAF','SSD','SDN',
             'TZA','TGO','TUN','UGA','ZMB','ZWE']

# Dict of countries names with ISO3 codes as keys
countries_names_dict = {
'DZA' : 'Algeria',
'AGO' : 'Angola',
'BEN' : 'Benin',
'BWA' : 'Botswana',
'BFA' : 'Burkina Faso',
'BDI' : 'Burundi',
'CMR' : 'Cameroon',
'CAF' : 'Central African Republic',
'TCD' : 'Chad',
'COD' : 'Democratic Republic of the Congo',
'COG' : 'Congo',
'CIV' : 'Cote d\'Ivoire',
'DJI' : 'Djibouti',
'EGY' : 'Egypt',
'GNQ' : 'Equatorial Guinea',
'ERI' : 'Eritrea',
'SWZ' : 'Eswatini',
'ETH' : 'Ethiopia',
'GAB' : 'Gabon',
'GMB' : 'Gambia',
'GHA' : 'Ghana',
'GIN' : 'Guinea',
'GNB' : 'Guinea Bissau',
'KEN' : 'Kenya',
'LSO' : 'Lesotho',
'LBR' : 'Liberia',
'LBY' : 'Libya',
'MWI' : 'Malawi',
'MLI' : 'Mali',
'MRT' : 'Mauritania',
'MAR' : 'Morocco',
'MOZ' : 'Mozambique',
'NAM' : 'Namibia',
'NER' : 'Niger',
'NGA' : 'Nigeria',
'RWA' : 'Rwanda',
'SEN' : 'Senegal',
'SLE' : 'Sierra Leone',
'SOM' : 'Somalia',
'ZAF' : 'South Africa',
'SSD' : 'South Sudan',
'SDN' : 'Sudan',
'TZA' : 'United Republic of Tanzania',
'TGO' : 'Togo',
'TUN' : 'Tunisia',
'UGA' : 'Uganda',
'ZMB' : 'Zambia',
'ZWE' : 'Zimbabwe'
}

# Dict of columns names and full variable names
parameters_columns_full_names = {'dg' : 'Dog population',
                                 'dgi' : 'Dog pop. annual increase',
                                 'vp24' : 'Vaccine price 2024',
                                 'pepp24' : 'PEP price 2024',
                                 'dist_probpep' : 'Probability receiving PEP',
                                 'E_all' : 'Exposure factor',
                                 'probcc_all' : 'Clinical case probability',
                                 'pepdemand_all' : 'PEP demand factor'}

parameters_columns_names_list = ['country_code', 'dg', 'dgi', 'vp24', 'pepp24', 'probpep', 'E',
                                                  'probcc', 'pepdemand']

parameters_columns_full_names_list = ['Dog population', 'Dog pop. annual increase', 'Vaccine price 2024',
                                 'PEP price 2024', 'Probability receiving PEP', 'Exposure factor',
                                 'Clinical case probability ', 'PEP demand factor']

# List of columns for summary results
summary_columns = ['country_code',
                   'gain_loss_to_baseline_0025',
                   'gain_loss_to_baseline_05',
                   'gain_loss_to_baseline_mean',
                   'gain_loss_to_baseline_0975',
                   'gain_loss_to_baseline_0025_log',
                   'gain_loss_to_baseline_05_log',
                   'gain_loss_to_baseline_mean_log',
                   'gain_loss_to_baseline_0975_log',
                   'strategy_payoff_0025',
                   'strategy_payoff_05',
                   'strategy_payoff_mean',
                   'strategy_payoff_0975',
                   'baseline_payoff_0025',
                   'baseline_payoff_05',
                   'baseline_payoff_mean',
                   'baseline_payoff_0975',
                   'number_of_loss_to_baseline',
                   'percentage_losses'
                   ]

# List of countries with PEP as dominant strategy
coalition_pep = ['BWA', 'DZA', 'EGY', 'GAB', 'LBY', 'NAM', 'TUN', 'ZAF']


##################### Functions #########################

def create_and_save_distance_matrix():
    # Read the adjacency matrix of African countries
    neighbouring_countries = pd.read_excel("../data/Appendix 7 - Neighbouring countries matrix.xlsx", header=0, index_col=0)
    neighbouring_countries.fillna(value = 0, inplace = True)
    neighbouring_countries = neighbouring_countries-np.eye(48)

    # Creating the graph
    G = nx.from_numpy_array(neighbouring_countries.values)

    # Computing the shortest distances between pairs of countries
    length = dict(nx.all_pairs_shortest_path_length(G))

    # Creating a DataFrame with the distanes
    distance_matrix = pd.DataFrame.from_dict(length).sort_index(ascending=True)
    distance_matrix = distance_matrix + 10*np.eye(48)

    # Saving the DataFrame
    distance_matrix.to_csv("distance_matrix.csv", encoding='UTF-8', index=True, header=True, sep='\t')

def read_all_data(neighbour_countries_file = "Appendix 7 - Neighbouring countries matrix.xlsx", #Neighbour countries.xlsx
                  distance_matrix_file = "Appendix 8 - Minimal distance matrix.csv", #distance_matrix.csv
                  data_for_simulation_file = "Appendix 4 - Data for simulations.xlsx"): #summary_data_for_simulation.xlsx

    # Read the file with the adjacency matrix of countries
    neighbouring_countries = pd.read_excel(f"../data/{neighbour_countries_file}", header=0, index_col=0)
    neighbouring_countries.fillna(value = 0, inplace = True)
    neighbouring_countries = neighbouring_countries-np.eye(48)

    # Read the file with the distance matrix
    distance_matrix = pd.read_csv(f"../data/{distance_matrix_file}", encoding='UTF-8', index_col=0, header=0, sep='\t')

    # Read the file with parameters to use for the simulations
    df = pd.read_excel(f"../data/{data_for_simulation_file}", #./Alvar/Dog Population Knobel 2005 including PopGrowth.xlsx
                       sheet_name="Pop_Data",
                       header=0)

    # Read the file with GDP information to use for the simulations
    gdp_contribution = pd.read_excel(f"../data/{data_for_simulation_file}",
                                     sheet_name = "AVG_HCE_Data",
                                     header = 0)

    return df, neighbouring_countries, distance_matrix, gdp_contribution

# Get the strategy profile from the folder name
def experiment_strategy_from_folder(folder = '../results/s_vaccination'):
    if 's_vaccination' in folder:
        return 'all_vac'
    elif 's_reintroduction' in folder:
        return  'one_vac'
    else:
        raise ValueError('Unknown folder!')

# Transform the week number to year
def transform_week_to_year(week):
    year = 2024 + (week-1)//52
    return year

# Create the discounted prices
def create_discounted_vector(price, number_years=31):

    if (type(price) == int) or (type(price) == float) :
        vector = np.array([price/np.power(1.05,i) for i in range(number_years)])
        vector = vector.reshape((-1, 1))
    elif (len(price==48)):
        discount_vector = np.array([1/np.power(1.05,i) for i in range(number_years)]).reshape((-1,1))
        vector = discount_vector * price.values.reshape((1,-1))
    else : vector = 0

    return vector

# Special log transformation
def log_transform(row):
    if row > 1:
        return np.log10(row)
    elif row < 1:
        return -np.log10(-row)
    else : return 0

def odes(state:List[float], t:np.array, pars:List[float]):
    """
    This function odes represents the ODE system that governs the dynamics of a disease transmission model.
    The model considers the susceptible individuals (S), the exposed individuals (E), the infectious individuals (I),
    the vaccinated individuals (V), the cumulative infected individuals (C), and the cumulative vaccinated individuals (Vc).

    :param state: Initial conditions for the different population
    :param t: Timesteps
    :param pars: List of parameters' values
    :return: Solutions for all population as a list
    """

    # Reading the parameters
    N0, beta, mu, sigma, delta, nu, alpha, epsilon, gamma, strategy, week_of_reintroduction = pars # distance_to_reservoir à rempalcer par week_of_reintroduction ?
    S, E, I, V, C, Vc = state
    epsilon_save = epsilon.copy()

    # Definition of strategy dependent values
    if strategy==0:pass
    elif strategy==1 :
        if (t >= 0 and t < 105) : alpha = 0.024
        if (t >= 105): alpha = 0
        if week_of_reintroduction==0:
            epsilon = 0
        else:
            if (t >= 0 and t < week_of_reintroduction): epsilon = 0
            if (t >= week_of_reintroduction): epsilon = epsilon_save
    else: raise ValueError('There is an invalid strategy! (Only 0 and 1 are allowed)')

    # Definition of ODEs
    dydt = [
        mu * N0 * np.exp(gamma * t) - (nu * alpha + mu) * S - beta * S * I / (S + E + I + V), # Susceptible
        beta * S * I / (S + E + I + V) - (sigma + mu) * E + epsilon * np.exp(gamma * t), # Exposed
        sigma * E - (delta + mu) * I, # Infected
        nu * alpha * S - mu * V, # Vaccinated
        sigma * E, # Infected cumulative
        nu * alpha * S # Vaccinated cumulative

    ]

    return  dydt #[dS, dE, dI, dV, dC]


def create_bash_script(type_script,
                       country_code,
                       number_sim=500000,
                       reintroduction=0,
                       save_sim_results = 1,
                       min_dist_to_compare=1,
                       cpu_per_task=4,
                       mem_per_cpu=32):
    """
    Function that generates slurm job for running computations on scicore cluster.

    :param type_script: Selecting the type of analysis we want to perform. Valid values are: 'sensitivity_analysis' for sensitivity analysis "
                         at country level; 'monte_carlo' for estimating rabid dog populations by Monte Carlo simulations.
    :param country_code: Selecting the country we want to perform analysis for.
    :param number_sim: Number of simulations to run. It corresponds to the parameter "size" in openturns experiments.
    :param reintroduction: Run the sensitivity analysis with reintroduction. 1 : True, 0 : False
    :param save_sim_results: Save the results of the simulations for the sensitivity analysis. 1 : True, 0 : False
    :param min_dist_to_compare: The distance of the closest infected country in the case of reintroduction.
    :param cpu_per_task: Number of CPUs to use for calculation. Unfortunately, the solving of ODEs in the script is sequential,
            it will not affect much the computational speed.
    :param mem_per_cpu: Memory available for computation.
    :return:
    """

    if type_script == 'sensitivity_analysis':
        script_name = 'sensitivity_analysis_model_one_country.py'
        b_sensitivity = True

    elif type_script == 'monte_carlo':
        script_name = 'monte_carlo_baseline_populations.py'
        reintroduction = 0
        b_sensitivity = False

    else:
        raise ValueError("Wrong type_script parameter! Valid values are: 'sensitivity_analysis' for sensitivity analysis "
                         "at country level; 'monte_carlo' for estimating rabid dog populations by Monte Carlo simulations.")



    res = f"""#!/bin/bash
            #The previous line is mandatory
            
            #SBATCH --job-name={type_script}_rabies_{country_code}     #Name of your job
            #SBATCH --cpus-per-task={cpu_per_task}    #Number of cores to reserve
            #SBATCH --mem-per-cpu={mem_per_cpu}G     #Amount of RAM/core to reserve
            #SBATCH --time=1-00:00:00      #Maximum allocated time
            #SBATCH --qos=1day         #Selected queue to allocate your job
            #SBATCH --output=myrun.o%j   #Path and name to the file for the STDOUT
            #SBATCH --error=myrun.e%j    #Path and name to the file for the STDERR
            
            module load Python/3.9.5-GCCcore-10.3.0       #Load required modules
            source $HOME/ot_venv/bin/activate
            python {script_name} {number_sim} {country_code} {reintroduction if b_sensitivity else ''}  {save_sim_results if b_sensitivity else ''} {min_dist_to_compare if b_sensitivity else ''}#Execute your command(s)"""

    with open('../jobs/slurm_job_{}{}{}_{}.sh'.format(type_script,
                                                           '_reintroduction_' if reintroduction == 1 else '' ,
                                                            f'dist{min_dist_to_compare}' if reintroduction == 1 else '' ,
                                                            country_code), 'w', newline='\n') as f:
        f.write(res)


if __name__=='__main__':

    # Creating scripts for slurm for sensitivity analysis
    for country_code in countries_codes_list:
        create_bash_script('sensitivity_analysis',
                           country_code,
                           number_sim=20000,
                           reintroduction=0,
                           save_sim_results=1,
                           min_dist_to_compare=1,
                           cpu_per_task=4,
                           mem_per_cpu=16)

        create_bash_script('sensitivity_analysis',
                           country_code,
                           number_sim=20000,
                           reintroduction=1,
                           save_sim_results=1,
                           min_dist_to_compare=1,
                           cpu_per_task=4,
                           mem_per_cpu=16)