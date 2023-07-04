# Imports
import logging

import pandas as pd
import numpy as np
from scipy.integrate import odeint
import openturns as ot
import time
import warnings
import os
import sys
from typing import List

# Imports from utils.py
from utils import read_all_data, transform_week_to_year, odes

# Options
pd.set_option("display.expand_frame_repr", False)
warnings.filterwarnings("ignore")

# Creation of results folders
outdirs = ['../results', '../results/dog_population']
for outdir in outdirs:
    if not os.path.exists(outdir):
        os.mkdir(outdir)


class monte_carlo_baseline_populations(ot.OpenTURNSPythonFunction) :
    def __init__(self, numberInputs:int = 4, numberOutputs:int = 1) -> None:
        """
        Constructor for initializing the class for the Monte-Carlo simulations. This function initializes the class by setting the input and output descriptions,
        reading data from files, initializing variables for simulation and creating an empty
        pandas dataframe for storing simulation results.

        :param numberInputs: Number of input parameters
        :param numberOutputs: Number of output parameters
        """

        # Create an OpenTURNS function based on a Python function
        ot.OpenTURNSPythonFunction.__init__(self, numberInputs, numberOutputs)

        # Description of inputs and outputs
        self.setInputDescription(['dg', 'dgi', 'E_all', 'probcc_all'])
        self.setOutputDescription(['rabid_dogs'])

        # Reading and saving data as attributes of the class
        self.df, self.neighbouring_countries, self.distance_matrix, self.df_gdp = read_all_data()

        # Initializing the total number of executed simulations
        self.num_sim = 0

        # Defining the minimal distance to the closest infected country as 0 (no reintroduction or infected)
        self.reintroduction = False
        self.min_dist = 0

        # Initialize the adtaframe for saving the results
        self.df_res = pd.DataFrame(columns = ['country_code', 'rabid_dogs_population', 'exposed_humans',
                                                               'clinical_cases'])

    def select_country_and_create_dist(self, country_code:str, plot:bool = False) -> (ot.ComposedDistribution, List[str]):
        """
        Method to select a country by country code and create a composed distribution using data from that country.

        :param country_code: the country code to select.
        :param plot: whether to plot the created distribution or not.

        :returns: (dist, dist_names): dist: the composed distribution created using data from the selected country.
        dist_names: the names of the distributions used to create the composed distribution.
        """

        # Select data from the desired country
        self.country_code = country_code
        self.df = self.df[self.df['country_code']==country_code]

        # Create the composed distribution using data from the selected country
        dist, dist_names = create_composed_dist_for_sim_one_country(self.df,
                                                                    country_code)

        return dist, dist_names

    def select_reintroduction_scenario(self, dist:int = 1) -> None:
        """
        Method to select the reintroduction scenario based on the distance to the closest infected country.

        :param dist: the country code to select.

        :returns:
        """
        self.reintroduction = True
        self.min_dist = dist

    def save_resulting_dataframe(self) -> None:
        """
        Saving th results as dataframe. The resulting dataframe is saved as CSV file in UTF-8 encoding with ";" as
        separator and decimals option ".".

        :returns:
        """

        # Saving th results as dataframe
        self.df_res.to_csv('../results/dog_population/simulation_dog_population_results_{}.csv'.format(self.country_code), #, datetime.now().strftime("%Y%m%d_%H%M%S")
                           encoding='UTF-8',
                           sep=';',
                           decimal='.')

    def _exec(self, Y):
        """
        This function takes in a list of 4 parameters (Y) and executes a simulation for a single country using these parameters.
        It then appends the results of the simulation to the instance variable 'df_res' and returns the simulation output.

        :param Y: List of 4 parameters
        :return: rabid_dogs: The cumulative number of rabid dogs
        """

        # Reading the values in the input
        X = []
        for i in range(4):
            X.append(Y[i])

        # # Verbose for following
        if self.num_sim%1000==0: logging.debug(f'working: {self.num_sim}')

        parameters = ['dog_population_mean',  # name of the dog population vector
                      'dog_population_increase_mean',  # name of the dog population increase vector
                      'probability_pep',  # Probability of receiving PEP
                      'vac_price_2024_mean',  # Vaccination price for 2024
                      'vac_price_2024_mean',  # Vaccination price for 2025
                      'pep_price_2024_mean'  # PEP price in 2024
                      ]

        # Create the input data for computation
        df =self.df.copy()

        df['dog_population_mean'] = X[0]
        df['dog_population_increase_mean'] = X[1]
        dist_E, dist_cc = X[2], X[3]

        # Compute the result
        rabid_dogs, exposed_humans, clinical_cases = dog_population_calculation_one_country(df,
                                                                                            self.df_gdp,
                                                                                            parameters,
                                                                                            s_vec=0,
                                                                                            country_code=self.country_code,
                                                                                            min_dist=0,
                                                                                            cc_parameter=dist_cc,
                                                                                            exposure_parameter=dist_E,
                                                                                            pep_need_parameter=1.5
                                                                                            )

        df_res = [country_code, rabid_dogs, exposed_humans, clinical_cases]

        # Add the result to the resulting DataFrame
        self.df_res = pd.concat([self.df_res, pd.DataFrame(data =[df_res], columns=['country_code', 'rabid_dogs_population', 'exposed_humans',
                                                               'clinical_cases'])],
                                ignore_index=True)

        # Increase the simulation counter
        self.num_sim+=1

        return [rabid_dogs]

def create_composed_dist_for_sim_one_country(df:pd.DataFrame,
                                             country_code:str = 'DZA') -> (ot.ComposedDistribution, List[str]):
    """
    Create a composed distribution object and the list of distribution names for a specific country for the calculation
    of the rabid dog population, exposed humans and clinical cases.

    :param df: pandas DataFrame with all parameter values for different countries.
    :param country_code: the code for the country to simulate. Default is 'DZA'.

    :return: (distribution, all_dist_names): distribution: the composed distribution object.
    all_dist_names: list of distribution names.
    """

    # Filtering the data on the needed country
    df = df[df['country_code']==country_code]

    # Creating distributions for different parameters
    dist_dp = ot.BetaMuSigma(df['dog_population_mean'].values[0],
                          0.65 * (df['dog_population_ub'].values[0] -df['dog_population_lb'].values[0]) / 3.92,
                          df['dog_population_lb'].values[0],
                          df['dog_population_ub'].values[0]).getDistribution()

    dist_dpi = ot.BetaMuSigma(df['dog_population_increase_mean'].values[0],
                          0.6 * (df['dog_population_increase_ub'].values[0] -df['dog_population_increase_lb'].values[0]) / 3.92,
                          df['dog_population_increase_lb'].values[0],
                          df['dog_population_increase_ub'].values[0]).getDistribution()

    
    dist_E = ot.BetaMuSigma(2.3, 0.5, 1.5, 4.0).getDistribution()
    dist_cc = ot.Normal(0.19, 0.01)

    # Create the list of distribution
    all_dist = [dist_dp, dist_dpi, dist_E, dist_cc] # 4

    # Create the list with the names of variables
    all_dist_names =['dg', 'dgi', 'E_all', 'probcc_all']

    # Define a copula for the distributions
    copula = ot.IndependentCopula(4)

    # Create a composed distribution
    distribution = ot.ComposedDistribution(all_dist, copula)

    return distribution, all_dist_names

def dog_population_calculation_one_country(df:pd.DataFrame,
                                           df_gdp:pd.DataFrame,
                                           parameters:List[str],
                                           s_vec: int,
                                           country_code:str ='DZA',
                                           min_dist:int = 0,
                                           cc_parameter:float = 0.19,
                                           exposure_parameter:float = 2.3,
                                           pep_need_parameter:float = 1.5
                                           ):
    """
    Calculation of different population. It is a truncated version of the payoff calculation.

    :param df: DataFrame with parameters' values for calculation
    :param df_gdp: DataFrame with foregone GDP per capita estimations
    :param parameters: Parameters' columns to use
    :param s_vec: Value of the used strategy
    :param country_code: ISO3 country code of the selected country
    :param min_dist: Distance to the closest infected country
    :param cc_parameter: Value of the probability of developing a clinical case
    :param exposure_parameter: Number of dogs bite per dog
    :param pep_need_parameter: A multiplication factor for the PEP administration
    :return: (rabid_dogs, exposed_humans, clinical_cases, total_dog_population): rabid_dogs: Cumulative rabid dogs population.
    exposed_humans: Cumulative number of exposed humans. clinical_cases: Cumulative number of developed clinical cases.

    """

    # Filtering on the selected country
    df_inter = df.copy()
    df_inter = df_inter[df_inter['country_code']==country_code]

    df_gdp = df_gdp[['Year', country_code]]

    # Calculating the week of reintroduction based on the distance to the closest infected country
    week_of_reintroduction = 104*int(min_dist!=0) + 39 * min_dist

    # Adding parameters to the DataFrame
    df_inter['strategies'] = s_vec
    df_inter['min_dist_to_reservoir'] = min_dist
    df_inter['week_of_reintroduction'] = week_of_reintroduction

    # Create resulting DataFrames
    rabid_dog_population_sim = pd.DataFrame(data={'Week': range(1613)})
    vaccinated_dog_population_sim = pd.DataFrame(data={'Week': range(1613)})
    total_population_sim = pd.DataFrame(data={'Week': range(1613)})

    # Definition of ODEs parameters
    PopSize = df_inter[parameters[0]].values[0]  # Dog population size 4="Total.Dog.Population.Average"
    beta0 = 1.0319227995  # model fit to achieve I = 1 - 0.0108074632*105
    I0 = 0.57
    mu0 = 0.0066
    annualincrease = df_inter[parameters[1]].values[0] #7 = = "Annual.Increase.Average"
    # stable dog population in the absence of disease
    N0 = PopSize
    # transmission rate
    beta = beta0
    # birth/death rate
    mu = mu0
    # rate of progression from exposed to infectious state
    sigma = 0.239
    # disease induced death rate
    delta = 1.23
    # vaccination efficacy
    nu = 0.95
    # vaccination rate
    alpha = 0
    # rate of dog birth (epsilon)
    epsilon = 0.1362 / 30000 * PopSize
    # weekly population growth rate
    gamma = (1 / 52) * np.log(1 + annualincrease)

    strategy = s_vec
    weeks_before_reintroduction = df_inter['week_of_reintroduction'].values[0]

    pars = [
            N0,
            beta,
            mu,
            sigma,
            delta,
            nu,
            alpha,
            epsilon,
            gamma,
            strategy,
            weeks_before_reintroduction
        ]
    #print(pars)

    S = 0.9970297 * PopSize  # Number of succeptible dogs in patch k at time t - Susceptible
    E = 0.00009830711 * PopSize  # Number of exposed dogs in patch k at time t - Exposed
    I = 0.000019 * PopSize  # Number of rabied dogs in patch k at time t) - Infective 0.57 Laager model
    V = 0  # Number of vaccinated dogs in patch k at time t - Vaccinated
    C = 0  # cumulative number of rabid dogs
    Vc = 0
    initial_state = [S, E, I, V, C, Vc]

    times = np.arange(0, 1613, 1)

    # Computing the solution of ODEs
    out = np.array(odeint(odes, y0=initial_state, t=times, args=(pars,)))

    rabid_dog_population_sim = pd.concat([rabid_dog_population_sim, pd.Series(data = out[:,4], name = "{}".format(country_code))], axis = 1)
    vaccinated_dog_population_sim = pd.concat([vaccinated_dog_population_sim, pd.Series(data = out[:,5], name = "{}".format(country_code))], axis = 1)
    total_population_sim = pd.concat([total_population_sim, pd.Series(data = out[:,0] + out[:,1] + out[:,2] + out[:,3] , name = "{}".format(country_code))], axis = 1)

    # Rabid dog population
    rabid_dog_population_sim = rabid_dog_population_sim.diff()
    rabid_dog_population_sim = rabid_dog_population_sim.iloc[1:,:]
    rabid_dog_population_sim['Week'] = range(1, len(rabid_dog_population_sim)+1)
    rabid_dog_population_sim['Year'] = rabid_dog_population_sim['Week'].apply(lambda row : transform_week_to_year(row))

    rabid_dog_population_sim_year = rabid_dog_population_sim.groupby("Year", as_index= False).sum()
    rabid_dog_population_sim_year.drop(columns = ["Week", "Year"], axis = 1, inplace= True)

    # Exposed humans
    exposed_humans_sim_year = rabid_dog_population_sim_year*exposure_parameter

    # Clinical cases
    clinical_cases_humans_sim_year = exposed_humans_sim_year * cc_parameter

    return rabid_dog_population_sim_year.sum().values[0], exposed_humans_sim_year.sum().values[0], clinical_cases_humans_sim_year.sum().values[0]


def run_monte_carlo_populations(country_code, size):
    """
    Run Monte-Carlo simulations. For confidence intervals estimation, the output file is post-treated.

    :param country_code: Country code of the country we want to run Monte Carlo simulations
    :param size: Size of the sample for the sensitivity analysis
    :return:
    """

    st = time.time()

    # Initialization of the Monte Carlo class
    monte_carlo_population_run = monte_carlo_baseline_populations()

    # Creation of distribution
    distribution, all_dist_names = monte_carlo_population_run.select_country_and_create_dist(
        country_code, plot=False)

    # Definition of the LHS experiment for exploration of the parameter space
    lhs_experiment = ot.LHSExperiment(distribution, size)

    # Generation of the parameters vectors for the experiment
    inputDesign = lhs_experiment.generate()

    print("========================== Monte Carlo simulations (dog populations) for {} ============================".format(country_code))
    print(f'Number of simulations to run: {inputDesign.getSize()}')

    # Generate the model
    modele = ot.Function(monte_carlo_population_run)

    # Calculate the results of the inputDesign
    outputDesign = modele(inputDesign)

    et = time.time()
    print(f'Execution time: {et - st} seconds')

    # Saving the resulting dataframe for post-treatment
    monte_carlo_population_run.save_resulting_dataframe()


if __name__=='__main__':
    # df, neighbouring_countries, distance_matrix, df_gdp = read_all_data()

    # Reading the parameters from bash during run
    # monte_carlo_baseline_populations.py {size} {country_code}
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    else:
        size = 200000

    if len(sys.argv) > 2:
        country_code =sys.argv[2]
    else:
        country_code = 'COG'

    # Run the Monte Carlo simulations
    run_monte_carlo_populations(country_code, size)




