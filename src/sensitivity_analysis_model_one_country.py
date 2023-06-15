# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import openturns as ot
from openturns.viewer import View
import time
import warnings
import os
import sys
from typing import Any, List

# Imports from utils.py
from utils import parameters_columns_full_names, countries_names_dict, \
    read_all_data, transform_week_to_year, create_discounted_vector, odes

# Options
pd.set_option("display.expand_frame_repr", False)
warnings.filterwarnings("ignore")

# Creation of results folders
outdirs = ['../results', '../results/s_vaccination', '../results/s_reintroduction', '../img', '../img/inputs_dist']
for outdir in outdirs:
    if not os.path.exists(outdir):
        os.mkdir(outdir)


class sensitivity_analysis_model_one_country(ot.OpenTURNSPythonFunction) :

    def __init__(self, numberInputs:int = 8, numberOutputs:int = 1) -> None:
        """
        Constructor for initializing the class for the sensitivity analysis and possible
        Monte-Carlo simulations. This function initializes the class by setting the input and output descriptions,
        reading data from files, initializing variables for simulation and creating an empty
        pandas dataframe for storing simulation results.

        :param numberInputs: Number of input parameters
        :param numberOutputs: Number of output parameters

        :returns:
        """

        # Create an OpenTURNS function based on a Python function
        ot.OpenTURNSPythonFunction.__init__(self, numberInputs, numberOutputs)

        # Description of inputs and outputs
        self.setInputDescription(['dg', 'dgi', 'vp24', 'pepp24', 'dist_probpep', 'E_all', 'probcc_all', 'pepdemand_all'])
        self.setOutputDescription(['benefit_from_baseline'])

        # Reading and saving data as attributes of the class
        self.df, self.neighbouring_countries, self.distance_matrix, self.df_gdp = read_all_data()

        # Initializing the total number of executed simulations
        self.num_sim = 0

        # Defining the minimal distance to the closest infected country as 0 (no reintroduction or infected)
        self.min_dist = 0
        self.reintroduction = False

        # Initialize the adtaframe for saving the results
        self.df_res = pd.DataFrame(columns = ['country_code', 'benefice_from_baseline', 'probability_pep', 'vac_price_2024_mean',
                                  'dog_population_mean','dog_population_increase_mean','pep_price_2024_mean',
                                  'strategies', 'type_strategy', 'benefited_from_vac', 'cumulated_payoff', 'sim_id'])

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
                                                                    country_code,
                                                                    plot=plot)

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
        self.df_res.to_csv('{}/simulation_results_{}{}.csv'.format('../results/s_reintroduction' if self.reintroduction else '../results/s_vaccination',
                                                                    self.country_code,
                                                                    f'_dist_{self.min_dist}' if self.reintroduction else ''),
                           encoding='UTF-8',
                           sep=';',
                           decimal='.')

    def _exec(self, Y):
        """
        This function takes in a list of 8 parameters (Y) and executes a simulation for a single country using these parameters.
        It then appends the results of the simulation to the instance variable 'df_res' and returns the simulation output.

        :param Y: a list of 8 parameters

        :returns: output: The difference between the baseline payoff and the studied strategy payoff
        """

        # Reading the values in the input
        X = []
        for i in range(8):
            X.append(Y[i])

        # # Verbose for following
        # if self.num_sim%1000==0: print(f'working: {self.num_sim}')

        # Create the input data for computation
        df =self.df.copy()

        df['dog_population_mean'] = X[0]
        df['dog_population_increase_mean'] = X[1]
        df['vac_price_2024_mean']= X[2]
        df['pep_price_2024_mean'] = X[3]
        df['probability_pep'] = X[4]
        dist_E, dist_cc, dist_pepd = X[5], X[6], X[7]

        # Compute the result
        output, df_res = simulation_one_country(df,
                                                self.df_gdp,
                                                country_code = self.country_code,
                                                cc_parameter=dist_cc,
                                                exposure_parameter=dist_E,
                                                pep_need_parameter=dist_pepd,
                                                s_vec_to_compare=1,  # Vaccination strategy
                                                min_dist_to_compare = self.min_dist,
                                                num_sim=self.num_sim,
                                                save = False)

        # Add the result to the resulting DataFrame
        self.df_res = self.df_res.append(pd.DataFrame(data =[df_res],
                                                      columns=['country_code', 'benefice_from_baseline', 'probability_pep', 'vac_price_2024_mean',
                                  'dog_population_mean','dog_population_increase_mean','pep_price_2024_mean',
                                  'strategies', 'type_strategy', 'benefited_from_vac', 'cumulated_payoff', 'sim_id']),
                                                      ignore_index = True)

        # Increase the simulation counter
        self.num_sim+=1

        return output

def create_composed_dist_for_sim_one_country(df:pd.DataFrame,
                                             country_code:str = 'DZA',
                                             plot:bool = False) -> (ot.ComposedDistribution, List[str]):
    """
    Create a composed distribution object and the list of distribution names for a specific country.
    The function filters the input dataframe by the given country code, then creates distributions for different parameters.
    If plot is True, the function plots the probability density function (PDF) for each parameter and saves the
    figure to a file.
    Finally, the function returns an object of type ot.ComposedDistribution which is a multivariate distribution of
    eight independent distributions with the same copula, and a list of parameter names.

    :param df: pandas DataFrame with all parameter values for different countries.
    :param country_code: the code for the country to simulate. Default is 'DZA'.
    :param plot: whether to plot the distributions. Default is False.

    :returns: (distribution, all_dist_names): distribution: the composed distribution object.
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


    dist_vp24 = ot.BetaMuSigma(df['vac_price_2024_mean'].values[0],
                          0.6 * (df['vac_price_2024_ub'].values[0] -df['vac_price_2024_lb'].values[0]) / 3.92,
                          df['vac_price_2024_lb'].values[0],
                          df['vac_price_2024_ub'].values[0]).getDistribution()

    dist_pepp24 = ot.BetaMuSigma(float(df['pep_price_2024_mean'].values[0]),
                          0.6 * (float(df['pep_price_2024_ub'].values[0]) - float(df['pep_price_2024_lb'].values[0])) / 3.92,
                          float(df['pep_price_2024_lb'].values[0]),
                          float(df['pep_price_2024_ub'].values[0])).getDistribution()

    # For high level of probability of receiving PEP, we reduce the variance
    if country_code in ['DZA', 'BWA', 'GAB', 'LBY', 'NAM', 'ZAF', 'TUN', 'EGY']:
        sigma = 0.06
    else: sigma = 0.25

    dist_probpep = ot.BetaMuSigma(df['probability_pep'].values[0],
                          sigma * (df['probability_pep_ub'].values[0] -df['probability_pep_lb'].values[0]) / 3.92,
                          df['probability_pep_lb'].values[0],
                          df['probability_pep_ub'].values[0]).getDistribution()

    
    dist_E = ot.BetaMuSigma(2.3, 0.5, 1.5, 4.0).getDistribution()
    dist_cc = ot.Normal(0.19, 0.01)
    dist_pepd = ot.Normal(1.5, 0.15)

    # Create the list of distribution
    all_dist = [dist_dp, dist_dpi, dist_vp24, dist_pepp24, dist_probpep, dist_E, dist_cc, dist_pepd] # 8

    # Create the list with the names of variables
    all_dist_names =['dg', 'dgi', 'vp24', 'pepp24', 'dist_probpep', 'E_all', 'probcc_all', 'pepdemand_all']

    # Plot all distributions
    if plot:
        all_intervals = [(0.9*df['dog_population_lb'].values[0], 1.05*df['dog_population_ub'].values[0]),
                         (0.9*df['dog_population_increase_lb'].values[0], 1.05*df['dog_population_increase_ub'].values[0]),
                         (0.9*df['vac_price_2024_lb'].values[0], 1.05*df['vac_price_2024_ub'].values[0]),
                         (0.9*float(df['pep_price_2024_lb'].values[0]), 1.05*float(df['pep_price_2024_ub'].values[0])),
                         (0.9*df['probability_pep_lb'].values[0], 1.05*df['probability_pep_ub'].values[0]),
                         (1, 5),
                         (0.15, 0.25),
                         (0.5, 2.5)
        ]
        pos = [(i, j) for i in range(2) for j in range(4)]
        grid = ot.GridLayout(2, 4)

        for i,dist in enumerate(all_dist):

            graph = dist.drawPDF(all_intervals[i][0],
                                 all_intervals[i][1],
                                 1000)

            graph.setTitle("{}".format(parameters_columns_full_names[all_dist_names[i]]))

            grid.setGraph(pos[i][0],
                          pos[i][1],
                          graph)

            grid.setTitle(f"Distributions of all parameters for the country : {countries_names_dict[country_code]} ({country_code})")

        view = View(grid, plot_kw={'color': 'blue'})
        plt.show()
        view.save(
            f'../img/inputs_dist/input_distribution_{country_code}.png'
        )

    # Define a copula for the distributions
    copula = ot.IndependentCopula(8)

    # Create a composed distribution
    distribution = ot.ComposedDistribution(all_dist, copula)

    return distribution, all_dist_names

def payoff_calculation_one_country(df:pd.DataFrame,
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
    Payoff calculation for a country given a particular strategy, and a particular set of parameters.

    :param df: DataFrame with parameters' values for calculation
    :param df_gdp: DataFrame with foregone GDP per capita estimations
    :param parameters: Parameters' columns to use
    :param s_vec: Value of the used strategy
    :param country_code: ISO3 country code of the selected country
    :param min_dist: Distance to the closest infected country
    :param cc_parameter: Value of the probability of developing a clinical case
    :param exposure_parameter: Number of dogs bite per dog
    :param pep_need_parameter: A multiplication factor for the PEP administration
    :return: total_payoff: Total payoff of the selected country (see methods for the formula)
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

    rabid_dog_population_sim = rabid_dog_population_sim.diff()
    rabid_dog_population_sim = rabid_dog_population_sim.iloc[1:,:]

    # Annual vaccinated population calculation
    vaccinated_dog_population_sim = vaccinated_dog_population_sim.diff()
    vaccinated_dog_population_sim = vaccinated_dog_population_sim.iloc[1:,:]
    vaccinated_dog_population_sim['Week'] = range(1, len(vaccinated_dog_population_sim)+1)
    vaccinated_dog_population_sim['Year'] = vaccinated_dog_population_sim['Week'].apply(lambda row : transform_week_to_year(row))
    vaccinated_dog_population_sim_year = vaccinated_dog_population_sim.groupby("Year", as_index=False).sum()
    vaccinated_dog_population_sim_year.drop(columns = ["Week", "Year"], axis=1, inplace=True)

    total_population_sim['Year'] = total_population_sim['Week'].apply(lambda row : transform_week_to_year(row))

    # Annual rabid population calculation
    rabid_dog_population_sim['Week'] = range(1, len(rabid_dog_population_sim)+1)
    rabid_dog_population_sim['Year'] = rabid_dog_population_sim['Week'].apply(lambda row : transform_week_to_year(row))
    rabid_dog_population_sim_year = rabid_dog_population_sim.groupby("Year", as_index= False).sum()
    rabid_dog_population_sim_year.drop(columns = ["Week", "Year"], axis = 1, inplace= True)

    # Exposure calculation
    exposed_humans_sim_year = rabid_dog_population_sim_year*exposure_parameter
    clinical_cases_humans_sim_year = exposed_humans_sim_year * cc_parameter
    pep_need_sim_year = exposed_humans_sim_year*pep_need_parameter

    # Payoffs calculation
    probability_pep = (1 - df_inter[parameters[2]].values).reshape((1, -1))
    payoff_hce_cost = -probability_pep * np.multiply(clinical_cases_humans_sim_year,
                                                         df_gdp.iloc[:, 1:])  # HCE Costs

    payoff_pep_cost = -pep_need_sim_year * create_discounted_vector(df_inter[parameters[5]], 31)

    payoff_vac_cost = vaccinated_dog_population_sim_year.copy()
    payoff_vac_cost.iloc[0,:] = - np.multiply(payoff_vac_cost.iloc[0,:].values, df_inter[parameters[3]].values.reshape(1,-1))
    payoff_vac_cost.iloc[1,:] = - np.multiply(payoff_vac_cost.iloc[1,:].values, df_inter[parameters[4]].values.reshape(1,-1))
    payoff_vac_cost.iloc[2,:] = - np.multiply(payoff_vac_cost.iloc[2,:].values, df_inter[parameters[4]].values.reshape(1,-1)/1.05)

    total_payoff = payoff_pep_cost + payoff_hce_cost + payoff_vac_cost
    total_payoff['Year'] = range(2024, 2055)

    df_inter.drop(columns = ["strategies", "min_dist_to_reservoir", "week_of_reintroduction"], inplace = True)

    return total_payoff

def simulation_one_country(df:pd.DataFrame,
                           df_gdp:pd.DataFrame,
                           country_code:str = 'DZA',
                           cc_parameter:float=0.19,
                           exposure_parameter:float=2.3,
                           pep_need_parameter:float=1.5,
                           s_vec_to_compare:int=1,
                           min_dist_to_compare:int = 0,
                           num_sim:int =0,
                           save:bool = False) -> (float, List[Any]):
    """
    Calculating the difference between the baseline strategy and another strategy for the same set of parameters.

    :param df: ataFrame with parameters' values for calculation
    :param df_gdp: DataFrame with foregone GDP per capita estimations
    :param country_code: ISO3 country code of the selected country
    :param cc_parameter: Value of the probability of developing a clinical case
    :param exposure_parameter: Number of dogs bite per dog
    :param pep_need_parameter: A multiplication factor for the PEP administration
    :param s_vec_to_compare: Value of the used strategy that we want to compare to the baseline (usually 1)
    :param min_dist_to_compare: The distance to the closest infected country for the reintroduction, for the strategy we want to compare
    :param num_sim: The simulation number
    :param save: Save results
    :return: (output, res_vec): output: The difference of total payoffs of the two strategies.
    res_vec: Resulting vector with different outputs from the comparison
    """

    # Filtering the data on the selected country
    df= df[df['country_code']==country_code]
    df_gdp = df_gdp[['Year', country_code]]


    parameters =['dog_population_mean', #name of the dog population vector
                     'dog_population_increase_mean', #name of the dog population increase vector
                     'probability_pep', #Probability of receiving PEP
                     'vac_price_2024_mean', #Vaccination price for 2024
                     'vac_price_2024_mean', #Vaccination price for 2025
                     'pep_price_2024_mean' #PEP price in 2024
                     ]

    # Calculation of the payoff for the baseline
    tp_1 = payoff_calculation_one_country(df,
                                          df_gdp,
                                          parameters= parameters,
                                          s_vec=0,
                                          country_code=country_code,
                                          min_dist=0,
                                          cc_parameter=cc_parameter,
                                          exposure_parameter=exposure_parameter,
                                          pep_need_parameter=pep_need_parameter,
                                          )


    # Calculation of the payoff of the strategy that we want to compare
    df_inter = df[['country_code','probability_pep','vac_price_2024_mean','dog_population_mean','dog_population_increase_mean','pep_price_2024_mean']].copy()
    tp_2 = payoff_calculation_one_country(df_inter,
                                          df_gdp,
                                          parameters= parameters,
                                          s_vec=s_vec_to_compare,
                                          country_code=country_code,
                                          min_dist=min_dist_to_compare,
                                          cc_parameter=cc_parameter,
                                          exposure_parameter=exposure_parameter,
                                          pep_need_parameter=pep_need_parameter,
                                          )

    # Difference
    delta = tp_2[country_code].sum() - tp_1[country_code].sum()

    # Results
    res_vec = [country_code,
               delta,
               df['probability_pep'].values[0],
               df['vac_price_2024_mean'].values[0],
               df['dog_population_mean'].values[0],
               df['dog_population_increase_mean'].values[0],
               df['pep_price_2024_mean'].values[0],
               s_vec_to_compare,
               's1_{}'.format('v' if min_dist_to_compare==0 else f'v_{min_dist_to_compare}'),
               int(delta>0),
               tp_2[country_code].sum(),
               f'sim_{num_sim}'
               ]


    output = res_vec[1], #ALL_VAC mean_delta s1_delta
                #res_vec[9], #ALL_VAC lost_from_vac s1_neg

    return output, res_vec

def run_sensitivity_analysis(country_code:str,
                             size:int,
                             reintroduction:bool = False,
                             min_dist_to_compare:int = 1,
                             save_simulation_results:bool = True):
    """
    Run the sensitivity analysis.

    :param country_code: Country code of the country we want to run the sensitivity analysis for
    :param size: Size of the sample for the sensitivity analysis
    :param reintroduction: Run the sensitivity analysis for the country with reintroduction.
    :param min_dist_to_compare: Distance to the closest infected country.
    :param save_simulation_results: Save the simulation results for the generated output as dataframe.

    :return:
    """

    st = time.time()

    # Initialization of the sensitivity analysis class
    sensitivity_analysis_model_one_country_run = sensitivity_analysis_model_one_country()

    # Creation of distribution
    distribution, all_dist_names = sensitivity_analysis_model_one_country_run.select_country_and_create_dist(
        country_code, plot=False)

    # Add the reintroduction if reintroduction is True
    if reintroduction:
        sensitivity_analysis_model_one_country_run.select_reintroduction_scenario(min_dist_to_compare)

    # Define the Sobol Incices Experiment
    sie = ot.SobolIndicesExperiment(distribution, size)

    # Generation of the parameters vectors for the experiment
    inputDesign = sie.generate()

    print("========================== Running sensitivity analysis for {} ============================".format(country_code))
    print(f'Number of simulations to run: {inputDesign.getSize()}')

    # Generate the model
    modele = ot.Function(sensitivity_analysis_model_one_country_run)

    # Calculate the results of the inputDesign
    outputDesign = modele(inputDesign)

    # Compute the Sobol Indices
    sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)

    et = time.time()
    print(f'Execution time: {et - st} seconds')

    output_dimension = modele.getOutputDimension()

    data = {}
    agg_first_order = [*sensitivityAnalysis.getAggregatedFirstOrderIndices()]
    agg_total_order = [*sensitivityAnalysis.getAggregatedTotalOrderIndices()]
    data[f'output_all_first_order'] = agg_first_order
    data[f'output_all_total_order'] = agg_total_order

    print("Agg. first order indices: ", agg_first_order)
    print("Agg. total order indices: ", agg_total_order)

    # Saving the resulting dataframe
    if save_simulation_results:
        sensitivity_analysis_model_one_country_run.save_resulting_dataframe()


    sensa_df = pd.DataFrame(data=data,
                            index=modele.getInputDescription())

    # Saving results for Sobol Indices
    sensa_df.to_csv('./results/{}/sensitivity_{}_{}.csv'.format('s_reintroduction' if reintroduction else 's_vaccination',
                                                                 country_code,
                                                                 'one_vac' if reintroduction else 'all_vac'),
                    encoding='UTF-8', sep=';')


if __name__=='__main__':
    # df, neighbouring_countries, distance_matrix, df_gdp = read_all_data()

    # Reading the parameters from bash during run
    # sensitivity_analysis_model_one_country.py {size} {country_code} {reintroduction} {save_simulation_results} {min_dist_to_compare}
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    else:
        size = 50000

    if len(sys.argv) > 2:
        country_code =sys.argv[2]
    else:
        country_code = 'COG'

    if len(sys.argv) > 3:
        reintroduction = False if int(sys.argv[3]) == 0 else True
    else:
        reintroduction = False

    if len(sys.argv) > 4:
        save_simulation_results = False if int(sys.argv[4]) == 0 else True
    else:
        save_simulation_results = True

    if len(sys.argv) > 5:
        min_dist_to_compare = int(sys.argv[5])
    else:
        min_dist_to_compare = 1


    # Run the sensitivity analysis
    run_sensitivity_analysis(country_code,
                             size,
                             reintroduction=reintroduction,
                             min_dist_to_compare=min_dist_to_compare,
                             save_simulation_results=save_simulation_results)




