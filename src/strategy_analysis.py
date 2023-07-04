# Imports
import logging

import pandas as pd
import numpy as np
import plotly.express as px
from random import randint, choices, sample
from scipy.integrate import odeint
from datetime import datetime
import warnings
import os
from typing import List

pd.set_option("display.expand_frame_repr", False)
warnings.filterwarnings("ignore")

# Imports from utils.py
from utils import parameters_columns_full_names, countries_names_dict, \
    read_all_data, transform_week_to_year, create_discounted_vector, odes

# Creation of results folders
outdirs = ['../results',  '../results/results_by_year/', '../results/s_reintroduction']
for outdir in outdirs:
    if not os.path.exists(outdir):
        os.mkdir(outdir)

def generate_strategy_vector(generation_type:str, number_vac:int = 10) -> np.ndarray:
    """
    Generate a strategy profile for all countries.

    :param generation_type: Type of the strategy profile to generate. Possible values : 'ALL_VAC', 'ALL_PEP', 'ONE_VAC',
    'RANDOM', 'NUMBER_VAC', 'NASH'.
    :param number_vac: Number of vaccinated countries if 'NUMBER_VAC' profile is selected.

    :return: strategy_profile: Resulting vector with chosen strategies for each country.
    """

    if generation_type=='ALL_VAC':
        return np.ones(48)

    elif generation_type=='ALL_PEP':
        return np.zeros(48)

    elif generation_type=='ONE_VAC':
        s_vec = np.zeros(48)
        pos_vac = randint(0, 47)
        s_vec[pos_vac] = 1

        return s_vec

    elif generation_type=='ONE_PEP':
        s_vec = np.ones(48)
        pos_pep = randint(0, 47)
        s_vec[pos_pep] = 0

        return s_vec
    
    elif generation_type=='RANDOM':
        return np.array(choices([0,1], k = 48))

    elif generation_type=='NUMBER_VAC':
        if number_vac >46 or number_vac < 2: raise ValueError("number_vac must be between 2 and 46")
        s_vec = np.zeros(48)
        s_vec[sample(range(48), number_vac)] = 1
        return s_vec
    elif generation_type=='NASH' :
        return np.array([
            0,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1
        ])
    else:
        raise ValueError("Please enter a valid parameter: \n Valid parameters are : ALL_VAC, ALL_PEP, ONE_VAC, ONE_PEP, RANDOM, NUMBER_VAC")

def parameter_generation(parameter_term:str) -> List[str]:
    """
    Generation of the list of columns to use as values.

    :param parameter_term:
    :return:
    """
    if parameter_term=='lb' :
        return ['dog_population_lb', #name of the dog population vector
                     'dog_population_increase_lb', #name of the dog population increase vector
                     'probability_pep', #Probability of receiving PEP
                     'vac_price_2024_mean', #Vaccination price for 2024
                     'vac_price_2024_mean', #Vaccination price for 2025
                     'pep_price_2024_mean' #PEP price in 2024
                     ]
    elif parameter_term=='ub':
        return ['dog_population_ub', #name of the dog population vector
                     'dog_population_increase_ub', #name of the dog population increase vector
                     'probability_pep', #Probability of receiving PEP
                     'vac_price_2024_mean', #Vaccination price for 2024
                     'vac_price_2024_mean', #Vaccination price for 2025
                     'pep_price_2024_mean' #PEP price in 2024
                     ]
    elif parameter_term=='default':
        return ['dog_population_mean',  # name of the dog population vector
         'dog_population_increase_mean',  # name of the dog population increase vector
         'probability_pep',  # Probability of receiving PEP
         'vac_price_2024_mean',  # Vaccination price for 2024
         'vac_price_2025_mean',  # Vaccination price for 2025
         'pep_price_2024_mean'  # PEP price in 2024
         ]
    else: pass

def strategy_experiment_generator(included_strategies = 'av'):
    """
    Generation of the experiment with needed number of simulation for different categories.

    :param included_strategies: av = ALL_VAC only, ovop = ALL_VAC + ONE_PEP (48) + ONE_VAC (48), mid = ovop + NUMBER_VAC_24 (271)
        all = mid + NUMBER_VAC_12 (271) +  NUMBER_VAC_36 (271)
    :return:
    """

    strategy_experiment_res = []

    s_vec_baseline = generate_strategy_vector("ALL_PEP")
    strategy_experiment_res.append((s_vec_baseline, "ALL_PEP"))

    s_vec_coop = generate_strategy_vector("ALL_VAC")
    strategy_experiment_res.append((s_vec_coop, "ALL_VAC"))

    s_vec_nash = generate_strategy_vector("NASH")
    strategy_experiment_res.append((s_vec_nash, "NASH"))

    if included_strategies in ['ovop', 'mid', 'all']:
        for i in range(48):
            s_vec = np.zeros(48)
            s_vec[i] = 1
            strategy_experiment_res.append((s_vec, 'ONE_VAC'))

        for i in range(48):
            s_vec = np.ones(48)
            s_vec[i] = 0
            strategy_experiment_res.append((s_vec, 'ONE_PEP'))

        if included_strategies in ['mid', 'all']:

            for i in range(271):
                s_vec = generate_strategy_vector("NUMBER_VAC", number_vac=24)
                strategy_experiment_res.append((s_vec, 'NUMBER_VAC_24'))

            if included_strategies in ['all']:

                for i in range(271):
                    s_vec = generate_strategy_vector("NUMBER_VAC", number_vac=12)
                    strategy_experiment_res.append((s_vec, 'NUMBER_VAC_12'))

                for i in range(271):
                    s_vec = generate_strategy_vector("NUMBER_VAC", number_vac=36)
                    strategy_experiment_res.append((s_vec, 'NUMBER_VAC_36'))

    return strategy_experiment_res

### Not used anymore
def simulation(df,
               df_gdp,
               distance_matrix,
               default=True,
               parameters_term = 'lb',
               num_sim = 0,
               cc_parameter=0.19,
               exposure_parameter=2.3,
               pep_need_parameter=1.5,
               decimals ='.',
               save = False,
               included_strategies = 'av'):
    """
    Calculating the difference between the baseline strategy and another strategy for the same set of parameters for all countries.
    """

    # Parameter generation
    if default:
        parameters =['dog_population_mean', #name of the dog population vector
                     'dog_population_increase_mean', #name of the dog population increase vector
                     'probability_pep', #Probability of receiving PEP
                     'vac_price_2024_mean', #Vaccination price for 2024
                     'vac_price_2025_mean', #Vaccination price for 2025
                     'pep_price_2024_mean' #PEP price in 2024
                     ]
    else:
        parameters = parameter_generation(parameters_term)

    strategy_experiment = strategy_experiment_generator(included_strategies)


    tp_1,tp_wo_hce_1, _, _, _ = payoff_calculation(df,
                                          df_gdp,
                                          distance_matrix,
                                          strategy_experiment[0][0],
                                          parameters,
                                          plot=False,
                                          strategy_profile_name="ALL_PEP",
                                          cc_parameter=cc_parameter,
                                          exposure_parameter=exposure_parameter,
                                          pep_need_parameter=pep_need_parameter,
                                          )

    res = pd.DataFrame(columns = ['country_code', 'benefice_from_baseline', 'probability_pep','vac_price_2024_mean','dog_population_mean','dog_population_increase_mean','pep_price_2024_mean',
                                  'strategies', 'type_strategy', 'benefited_from_vac',
                                  'cumulated_payoff', 'sim_id'])

    i=0
    n = len(strategy_experiment)

    for strategy_experiment_element in strategy_experiment:
        if i%100==0:
            print('{}/{} : {}'.format(i, n,strategy_experiment_element))

        df_inter = df[['country_code','probability_pep','vac_price_2024_mean','dog_population_mean','dog_population_increase_mean','pep_price_2024_mean']].copy()

        tp_2, tp_wo_hce_2, _, _, _  = payoff_calculation(df,
                                               df_gdp,
                                               distance_matrix,
                                               strategy_experiment_element[0],
                                               parameters,
                                               plot=False,
                                               strategy_profile_name=strategy_experiment_element[1])



        delta = tp_2 - tp_1

        delta.drop(columns = "Year", inplace = True)
        delta = delta.sum().to_frame('benefice_from_baseline')
        delta = delta.reset_index().rename(columns = {'index' : 'country_code'})

        df_inter['strategies'] = strategy_experiment_element[0]
        delta_T = pd.merge(delta, df_inter, on ='country_code', how = 'left')

        delta_T['type_strategy'] = strategy_experiment_element[1]

        delta_T['benefited_from_vac'] = (((delta_T['strategies']==1)&(delta_T['benefice_from_baseline']>0))).astype(int)

        tp_2.drop(columns = 'Year', inplace = True)
        tp_inter = pd.melt(tp_2, var_name="country_code", value_name="cumulated_payoff")
        tp_inter = tp_inter.groupby('country_code', as_index=False)['cumulated_payoff'].sum()

        delta_T = pd.merge(delta_T, tp_inter, right_on='country_code', left_on='country_code', how = 'left')
        delta_T['sim_id'] = f'sim_{i}'
        res = pd.concat([res, delta_T], axis=0)

        i+=1



    if save :
        res.to_csv('../results/strategy_analysis.csv',
                    encoding='UTF-8', sep=';', decimal=decimals)
        # for strategy_type in res['type_strategy'].unique():
        #     res[res['type_strategy']==strategy_type].to_csv('./results/strategy_{}_sim_{}.csv'.format(strategy_type, num_sim),
        #             encoding='UTF-8', sep=';', decimal=decimals)


    res['lost_from_vac'] = (res['benefice_from_baseline'] < 0).astype(int)

    output = [
                res.loc[res['type_strategy']=='ALL_VAC','benefice_from_baseline'].sum(), #ALL_VAC mean_delta s1_delta
                res.loc[res['type_strategy']=='ALL_VAC','lost_from_vac'].sum(), #ALL_VAC lost_from_vac s1_neg
            ]

    if included_strategies in ['ovop', 'mid', 'all']:
        output.append(res.loc[res['type_strategy']=='ONE_VAC','benefice_from_baseline'].sum()/48)
        output.append(res.loc[res['type_strategy']=='ONE_VAC','lost_from_vac'].sum()/48)
        output.append(res.loc[res['type_strategy']=='ONE_PEP','benefice_from_baseline'].sum()/48)
        output.append(res.loc[res['type_strategy']=='ONE_PEP','lost_from_vac'].sum()/48)

        if included_strategies in ['mid', 'all']:
            output.append(res.loc[res['type_strategy']=='NUMBER_VAC_24','benefice_from_baseline'].sum()/271)
            output.append(res.loc[res['type_strategy']=='NUMBER_VAC_24','lost_from_vac'].sum()/271)

            if included_strategies in ['all']:
                output.append(res.loc[res['type_strategy']=='NUMBER_VAC_12','benefice_from_baseline'].sum()/271)
                output.append(res.loc[res['type_strategy']=='NUMBER_VAC_12','lost_from_vac'].sum()/271)
                output.append(res.loc[res['type_strategy']=='NUMBER_VAC_36','benefice_from_baseline'].sum()/271)
                output.append(res.loc[res['type_strategy']=='NUMBER_VAC_36','lost_from_vac'].sum()/271)

    return output


def payoff_calculation(df:pd.DataFrame,
                       df_gdp:pd.DataFrame,
                       distance_matrix:pd.DataFrame,
                       s_vec:np.ndarray,
                       parameters:List[str],
                       cc_parameter:float = 0.19,
                       exposure_parameter:float = 2.3,
                       pep_need_parameter:float = 1.5,
                       plot:bool = False,
                       strategy_profile_name:str = 'ALL_VAC',
                       save_inter_results:bool = False
                       ):
    """
    Payoff calculation for all countries.

    :param df: DataFrame with parameters' values for calculation
    :param df_gdp: DataFrame with foregone GDP per capita estimations
    :param distance_matrix: Distance matrix between countries
    :param s_vec: Strategy profile
    :param parameters: Columns names to use as values
    :param cc_parameter: Value of the probability of developing a clinical case
    :param exposure_parameter: Number of dogs bite per dog
    :param pep_need_parameter: A multiplication factor for the PEP administration
    :param plot: Plot or not the payoffs
    :param strategy_profile_name: Strategy profile name
    :param save_inter_results: Save results by year for rabid dogs population, exposed humans, clinical cases and payoffs
    :return:
    """

    df_inter = df.copy()

    # Calculation of distances between vaccinated and closest non-vaccinated countries
    dist_matrix_with_strategies = np.multiply(
                                                np.multiply(
                                                    distance_matrix.values,
                                                    s_vec.reshape((-1,1))),
                                                1 - s_vec
                                                )

    dist_matrix_with_strategies[dist_matrix_with_strategies==0] = np.nan
    min_dist = np.nanmin(dist_matrix_with_strategies, axis=1)
    min_dist[np.isnan(min_dist)] = 0

    # Calculation of the week of reintroduction
    week_of_reintroduction = 104*(min_dist!=0).astype(int) + 39 * min_dist

    # Adding parameters to the DataFrame
    df_inter = pd.concat([df_inter,
                    pd.Series(data=s_vec, name="strategies"),
                    pd.Series(data=min_dist, name="min_dist_to_reservoir"),
                    pd.Series(data=week_of_reintroduction, name="week_of_reintroduction")],
                   axis = 1)

    # Create resulting DataFrames
    rabid_dog_population_sim = pd.DataFrame(data={'Week': range(1613)})
    vaccinated_dog_population_sim = pd.DataFrame(data={'Week': range(1613)})
    total_population_sim = pd.DataFrame(data={'Week': range(1613)})

    # Loop over the countries
    for i in range(48):

        # Definition of ODEs parameters
        PopSize = df_inter.at[i,parameters[0]]  # Dog population size 4="Total.Dog.Population.Average"
        beta0 = 1.0319227995  # model fit to achieve I = 1 - 0.0108074632*105
        I0 = 0.57
        mu0 = 0.0066
        annualincrease = df_inter.at[i, parameters[1]] #7 = = "Annual.Increase.Average"
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

        strategy = df_inter.at[i, 'strategies']
        weeks_before_reintroduction = df_inter.at[i, 'week_of_reintroduction']

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

        rabid_dog_population_sim = pd.concat([rabid_dog_population_sim, pd.Series(data = out[:,4], name = "{}".format(df_inter.iloc[i,0]))], axis = 1)
        vaccinated_dog_population_sim = pd.concat([vaccinated_dog_population_sim, pd.Series(data = out[:,5], name = "{}".format(df_inter.iloc[i,0]))], axis = 1)
        total_population_sim = pd.concat([total_population_sim, pd.Series(data = out[:,0] + out[:,1] + out[:,2] + out[:,3] , name = "{}".format(df_inter.iloc[i,0]))], axis = 1)

    rabid_dog_population_sim = rabid_dog_population_sim.diff()
    rabid_dog_population_sim = rabid_dog_population_sim.iloc[1:,:]

    # Annual vaccinated population calculation
    vaccinated_dog_population_sim = vaccinated_dog_population_sim.diff()
    vaccinated_dog_population_sim = vaccinated_dog_population_sim.iloc[1:,:]
    vaccinated_dog_population_sim['Week'] = range(1, len(vaccinated_dog_population_sim)+1)
    vaccinated_dog_population_sim['Year'] = vaccinated_dog_population_sim['Week'].apply(lambda row : transform_week_to_year(row))
    vaccinated_dog_population_sim_year = vaccinated_dog_population_sim.groupby("Year", as_index=False).sum()
    vaccinated_dog_population_sim_year.drop(columns = ["Week", "Year"], axis=1, inplace=True)
    # vaccinated_dog_population_sim_year.to_csv('./results/vaccinated_dogs_{}_{}.csv'.format(datetime.now().strftime("%Y%m%d"),datetime.now().strftime("%H%M%S")), encoding ='UTF-8', sep=';')

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

    payoff_pep_cost = -pep_need_sim_year * create_discounted_vector(df[parameters[5]], 31)

    payoff_vac_cost = vaccinated_dog_population_sim_year.copy()
    payoff_vac_cost.iloc[0,:] = - np.multiply(payoff_vac_cost.iloc[0,:].values, df_inter[parameters[3]].values.reshape(1,-1))
    payoff_vac_cost.iloc[1,:] = - np.multiply(payoff_vac_cost.iloc[1,:].values, df_inter[parameters[4]].values.reshape(1,-1))
    payoff_vac_cost.iloc[2,:] = - np.multiply(payoff_vac_cost.iloc[2,:].values, df_inter[parameters[4]].values.reshape(1,-1)/1.05)

    total_payoff = payoff_pep_cost + payoff_hce_cost + payoff_vac_cost
    total_payoff['Year'] = range(2024, 2055)

    payoff_without_hce = payoff_pep_cost + payoff_vac_cost

    # Saving the results
    if save_inter_results:
        file_name = f'n{s_vec.sum()}' if strategy_profile_name=="NUMBER_VAC" else strategy_profile_name.lower()

        rabid_dog_population_sim_year.to_csv(
            '../results/results_by_year/rabid_dog_population_{}.csv'.format(file_name),
            encoding='UTF-8', sep=';', decimal=".")

        clinical_cases_humans_sim_year.to_csv(
            '../results/results_by_year/clinical_cases_humans_{}.csv'.format(file_name),
            encoding='UTF-8', sep=';', decimal=".")

        exposed_humans_sim_year.to_csv(
            '../results/results_by_year/exposed_humans_{}.csv'.format(file_name),
            encoding='UTF-8', sep=';', decimal=".")

        total_payoff.to_csv(
            '../results/results_by_year/payoff_total_{}.csv'.format(file_name),
            encoding='UTF-8', sep=';', decimal=".")

        payoff_without_hce.to_csv(
            '../results/results_by_year/payoff_without_hce_{}.csv'.format(file_name),
            encoding='UTF-8', sep=';', decimal=".")

    # Plot the payoffs
    if plot:
        total_payoff_melted = pd.melt(total_payoff, id_vars="Year", var_name="country_code", value_name="payoff")
        total_payoff_melted['cumulated_payoff'] = total_payoff_melted.groupby(by='country_code')["payoff"].cumsum()
        total_payoff_melted = pd.merge(total_payoff_melted, df_inter[['country_code', 'strategies']], on='country_code', how = 'left')
        total_payoff_melted.to_csv(
        '../results/payoff_melted_total_{}_{}_{}.csv'.format(datetime.now().strftime("%Y%m%d"),
                                                            datetime.now().strftime("%H%M%S"),
                                                            strategy_profile_name),
            encoding='UTF-8', sep=';', decimal=",")

        fig_map = px.choropleth(total_payoff_melted,
                                title="Strategy : {}. Total payoff {:,} $".format(strategy_profile_name, int(total_payoff_melted['payoff'].sum())),
                                locations='country_code', color='cumulated_payoff',
                                hover_name='strategies', animation_frame="Year",
                                color_continuous_scale="reds_r",
                                range_color=[-1528118936,
                                             0]
                                )
        fig_map.update_layout(geo_scope="africa", geo_resolution=50)
        fig_map.show()

    df_inter.drop(columns = ["strategies", "min_dist_to_reservoir", "week_of_reintroduction"], inplace = True)

    return total_payoff, payoff_without_hce, rabid_dog_population_sim_year, clinical_cases_humans_sim_year, exposed_humans_sim_year


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)

    # Load data
    print("============================================= Strategy analysis script =============================================")
    print("Creating files with payoffs for some strategy profiles")

    df, neighbouring_countries, distance_matrix, df_gdp = read_all_data()

    logging.info('Data loaded')
    # Payoffs for all_vac strategy profile
    logging.info('Generating data for all_vac strategy profile')
    exp1 = 'ALL_VAC'
    payoff_calculation(df,
                       df_gdp,
                       distance_matrix,
                       generate_strategy_vector(exp1),
                       parameter_generation('default'),
                       save_inter_results= True,
                       strategy_profile_name= exp1
                       )

    # Payoffs for all_pep strategy profile
    logging.info('Generating data for all_pep strategy profile')
    exp1 = 'ALL_PEP'
    payoff_calculation(df,
                       df_gdp,
                       distance_matrix,
                       generate_strategy_vector(exp1),
                       parameter_generation('default'),
                       save_inter_results= True,
                       strategy_profile_name= exp1
                       )

    # Payoffs for nash strategy profile
    logging.info('Generating data for nash strategy profile')
    exp1 = 'NASH'
    payoff_calculation(df,
                       df_gdp,
                       distance_matrix,
                       generate_strategy_vector(exp1),
                       parameter_generation('default'),
                       save_inter_results= True,
                       strategy_profile_name= exp1
                       )


    exp2 = 'NUMBER_VAC_24'
    columns_names = ['DZA', 'AGO', 'BEN', 'BWA', 'BFA', 'BDI', 'CMR', 'CAF', 'TCD', 'COD', 'COG', 'CIV', 'DJI', 'EGY',
                       'GNQ', 'ERI', 'SWZ', 'ETH', 'GAB', 'GMB', 'GHA', 'GIN', 'GNB', 'KEN', 'LSO', 'LBR', 'LBY', 'MWI',
                       'MLI', 'MRT', 'MAR', 'MOZ', 'NAM', 'NER', 'NGA', 'RWA', 'SEN', 'SLE', 'SOM', 'ZAF', 'SSD', 'SDN',
                       'TZA', 'TGO', 'TUN', 'UGA', 'ZMB', 'ZWE', 'sim_id', 'year']

    total_payoff_samples = pd.DataFrame(columns=columns_names)
    payoff_without_hce_samples = pd.DataFrame(columns=columns_names)
    # rabid_dog_population_samples = pd.DataFrame(columns=columns_names)
    exposed_humans_samples = pd.DataFrame(columns=columns_names)

    # Payoffs for one_vac strategy profile for all countries
    logging.info('Generating data for one_pep strategy profile')

    for i in range(48):
        logging.debug("Simulation number: " + str(i))

        s_vec = np.ones(48)
        s_vec[i] = 0
        total_payoff, payoff_without_hce, rabid_dog_population_sim_year, \
        clinical_cases_humans_sim_year, exposed_humans_sim_year = payoff_calculation(df,
                                                              df_gdp,
                                                              distance_matrix,
                                                              s_vec,
                                                              parameter_generation('default'),
                                                              save_inter_results= False,
                                                              strategy_profile_name='ONE_PEP'
                                                              )

        total_payoff['sim_id'] = f'sim_{i}'
        payoff_without_hce['sim_id'] = f'sim_{i}'
        exposed_humans_sim_year['sim_id'] = f'sim_{i}'

        total_payoff['year'] = range(2024, 2055)
        payoff_without_hce['year'] = range(2024, 2055)
        exposed_humans_sim_year['year'] = range(2024, 2055)

        total_payoff_samples = pd.concat([total_payoff_samples, total_payoff], axis = 0)
        payoff_without_hce_samples = pd.concat([payoff_without_hce_samples, payoff_without_hce], axis = 0)
        exposed_humans_samples = pd.concat([exposed_humans_samples, exposed_humans_sim_year], axis = 0)

    total_payoff_samples.to_csv('../results/results_by_year/payoff_total_one_pep.csv', encoding='UTF-8', sep = ';', decimal ='.', index=True)
    payoff_without_hce_samples.to_csv('../results/results_by_year/payoff_without_hce_one_pep.csv', encoding='UTF-8', sep = ';', decimal ='.', index=True)
    exposed_humans_samples.to_csv('../results/results_by_year/exposed_humans_one_pep.csv', encoding='UTF-8', sep =';', decimal ='.', index=True)

    # Payoffs for NUMBER_VAC_24 strategy profile for a sample of 271 simulations
    total_payoff_samples = pd.DataFrame(columns=columns_names)
    payoff_without_hce_samples = pd.DataFrame(columns=columns_names)
    exposed_humans_samples = pd.DataFrame(columns=columns_names)

    logging.info('Generating data for n24 strategy profile')
    for i in range(271):

        logging.debug("Simulation number: " + str(i))
        total_payoff, payoff_without_hce, rabid_dog_population_sim_year, \
        clinical_cases_humans_sim_year, exposed_humans_sim_year = payoff_calculation(df,
                                                              df_gdp,
                                                              distance_matrix,
                                                              generate_strategy_vector('NUMBER_VAC', number_vac=24),
                                                              parameter_generation('default'),
                                                              save_inter_results= False,
                                                              strategy_profile_name='NUMBER_VAC_24'
                                                              )

        total_payoff['sim_id'] = f'sim_{i}'
        payoff_without_hce['sim_id'] = f'sim_{i}'
        exposed_humans_sim_year['sim_id'] = f'sim_{i}'

        total_payoff['year'] = range(2024, 2055)
        payoff_without_hce['year'] = range(2024, 2055)
        exposed_humans_sim_year['year'] = range(2024, 2055)

        total_payoff_samples = pd.concat([total_payoff_samples, total_payoff], axis=0)
        payoff_without_hce_samples = pd.concat([payoff_without_hce_samples, payoff_without_hce], axis=0)
        exposed_humans_samples = pd.concat([exposed_humans_samples, exposed_humans_sim_year], axis=0)

    total_payoff_samples.to_csv('../results/results_by_year/payoff_total_n24.csv', encoding='UTF-8', sep = ';', decimal ='.', index=True)
    payoff_without_hce_samples.to_csv('../results/results_by_year/payoff_without_hce_n24.csv', encoding='UTF-8', sep = ';', decimal ='.', index=True)
    exposed_humans_samples.to_csv('../results/results_by_year/exposed_humans_n24.csv', encoding='UTF-8', sep =';', decimal ='.', index=True)