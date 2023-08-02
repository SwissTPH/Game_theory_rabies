# Imports
import sys
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from scipy.stats import gaussian_kde
import warnings
import geopandas as gpd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import BoundaryNorm
import openturns as ot
import os
from shapely.geometry import Polygon
import logging

# Imports from utils.py
from utils import  countries_names_dict, parameters_columns_names_list,parameters_columns_full_names_list, \
    countries_codes_list, summary_columns, coalition_pep, log_transform, experiment_strategy_from_folder

# Options
pd.set_option("display.expand_frame_repr", False)
warnings.simplefilter(action='ignore', category=FutureWarning)
fmt = '${x:,.0f}'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

# Creation of results folders
outdirs = ['../results', '../results/s_vaccination', '../results/s_reintroduction', '../img', '../img/inputs_dist',
           '../img', '../img/all_vac', '../img/one_vac']

for outdir in outdirs:
    if not os.path.exists(outdir):
        os.mkdir(outdir)


def sobol_indices_compile_and_plot(folder:str = '../results/s_vaccination',
                                   save:bool = True,
                                   plot:bool = True) -> None:
    """
    Compile all Sobol Indices resulting from the country-level sensitivity analysis and plot using seaborn.

    :param folder: Folder where the results are saved (s_vaccination or s_reintroduction)
    :param save: Saving the file or not
    :param plot: Plot the indices or not
    :return:
    """

    logging.info("Starting sobol_indices_compile_and_plot function")

    # Get the strategy name from the folder name
    experiment_strategy = experiment_strategy_from_folder(folder)

    # Create DataFrames for saving the results
    all_first_sobol_indices = pd.DataFrame(columns=parameters_columns_names_list)
    all_total_sobol_indices = pd.DataFrame(columns=parameters_columns_names_list)

    logging.info("Starting loop over countries")
    # Loop over the countries to compile the data
    for country_code in countries_codes_list:
        logging.debug(f"Starting loop for country {country_code}")

        df = pd.read_csv(f'{folder}/sensitivity_{country_code}_{experiment_strategy}.csv',
                         encoding='UTF-8',
                         sep = ';',
                         decimal='.')

        df.rename(columns = {'output_all_first_order' : 'first_order',
                             'output_all_total_order' : 'total_order',
                             'Unnamed: 0' : 'variable'},
                  inplace = True)

        all_first_sobol_indices = pd.concat([all_first_sobol_indices,
                                             pd.DataFrame(
                                                 data=[np.concatenate(([country_code], df['first_order'].values))],
                                                 columns=parameters_columns_names_list)
                                             ],
                                            ignore_index = True)

        all_total_sobol_indices = pd.concat([all_total_sobol_indices,
                                             pd.DataFrame(
                                                 data=[np.concatenate(([country_code], df['total_order'].values))],
                                                 columns=parameters_columns_names_list)
                                             ],
                                            ignore_index = True)


    logging.info("Loop over countries finished")

    # Transforming values into float
    all_first_sobol_indices[parameters_columns_names_list[1:]] = all_first_sobol_indices[parameters_columns_names_list[1:]].astype(float)
    all_total_sobol_indices[parameters_columns_names_list[1:]] = all_total_sobol_indices[parameters_columns_names_list[1:]].astype(float)

    # Capping negative values at 0. Sobol indices are positive, the confidence interval is around zero and some values can appear negative
    all_first_sobol_indices[parameters_columns_names_list[1:]] = np.where(all_first_sobol_indices[parameters_columns_names_list[1:]] < 0, 0, all_first_sobol_indices[parameters_columns_names_list[1:]])
    all_total_sobol_indices[parameters_columns_names_list[1:]] = np.where(all_total_sobol_indices[parameters_columns_names_list[1:]] < 0, 0, all_total_sobol_indices[parameters_columns_names_list[1:]])

    # Saving the resulting DataFrames
    if save :
        logging.info("Saving the resulting DataFrames")
        all_first_sobol_indices.to_csv(f'../results/first_order_sobol_{experiment_strategy}.csv', encoding ='UTF-8', sep = ';', decimal='.')
        all_total_sobol_indices.to_csv(f'../results/total_order_sobol_{experiment_strategy}.csv', encoding ='UTF-8', sep = ';', decimal='.')

    # Ploting the Sobol indices
    if plot :
        logging.info("Plotting the Sobol indices")
        fig, (ax1, ax2) = plt.subplots(1,2)

        fig.suptitle(f'Sobol indices for {experiment_strategy} strategy')
        sns.heatmap(all_first_sobol_indices[parameters_columns_names_list[1:]],
                    xticklabels = parameters_columns_full_names_list,
                    yticklabels = all_first_sobol_indices['country_code'].values,
                    annot = True, fmt =  ".3f",
                    square=False, linewidths=.5, cbar_kws={"shrink": .5},
                    cmap = sns.light_palette("seagreen", as_cmap=True),
                    ax = ax1,
                    cbar = False,
                    vmin = 0.0,
                    vmax = 1.0)

        ax1.xaxis.tick_top()
        ax1.set(ylabel="Country code")
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="left", va ='bottom',
                 rotation_mode="anchor")
        plt.setp(ax1.get_yticklabels(), rotation=0)
        ax1.set_title("a) First order Sobol' indices")

        sns.heatmap(all_total_sobol_indices[parameters_columns_names_list[1:]],
                    xticklabels = parameters_columns_full_names_list,
                    yticklabels = "",
                    annot = True, fmt =  ".3f",
                    square=False, linewidths=.5, cbar_kws={"shrink": .5},
                    cmap = sns.light_palette("seagreen", as_cmap=True),
                    ax = ax2,
                    vmin = 0.0,
                    vmax = 1.0)
        ax2.xaxis.tick_top()
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="left", va ='bottom',
                 rotation_mode="anchor")
        ax2.set_title("b) Total order Sobol' indices")
        plt.show()

        fig.savefig(f'../img/sobol_indices_{experiment_strategy}.pdf',
                    dpi='figure',
                    format='pdf',
                    metadata=None,
                    bbox_inches=None,
                    pad_inches=0.1,
                    facecolor='auto',
                    edgecolor='auto',
                    backend=None)
    logging.info("Finished")

def data_compile_save_and_plot(folder:str = '../results/s_vaccination',
                                save_summary:bool = True,
                                save_compiled_data:bool = True,
                                plot_individual_results:bool = True):
    """
    Compile data from payoffs calculations (files generated during the sensitivity analysis) and summarize. Then, plot the data at individual level or all countries.

    :param folder: Folder containing the data.
    :param save_summary: Save the summary of results.
    :param save_compiled_data: Save the compiled data.
    :param plot_individual_results: Plot the individual results.
    :param plot_payoffs_comparing: Plot all results.
    :return:
    """
    logging.info("Starting data_compile_save_and_plot function")

    # Initializing resulting dataframes
    data = pd.DataFrame(columns=['country_code', 'benefice_from_baseline', 'cumulated_payoff', 'sim_id'])
    summary_data = pd.DataFrame(columns=summary_columns)

    # Get the strategy profile name from folder name
    experiment_strategy = experiment_strategy_from_folder(folder)

    logging.info(f"Strategy profile: {experiment_strategy}")
    logging.info("Starting the loop over the countries for compiling the results")

    # Loop over the countries for compiling the results
    for country_code in countries_codes_list:
        logging.info(f"Processing country code: {country_code}")
        df = pd.read_csv(f'{folder}/simulation_results_{country_code}_{experiment_strategy}.csv',
                         encoding='UTF-8',
                         sep = ';',
                         decimal='.')

        df['country_code'] = country_code
        df['baseline_payoff'] = df['cumulated_payoff'] - df['benefice_from_baseline']

        nbr_negative_ben = np.sum(df['benefice_from_baseline']<0)
        nombre_sim = len(df['benefice_from_baseline'])

        print('{} : Quantiles [2.5%, 97.5%] for gains/losses compared to baseline: [{:0f}, {:0f}]'.format(country_code, df['benefice_from_baseline'].quantile(0.025),
                                                                                                df['benefice_from_baseline'].quantile(0.975)))
        # Summary
        summary_data = pd.concat([summary_data, pd.DataFrame(data = [[
            f'{country_code}',
            df['benefice_from_baseline'].quantile(0.025),
            df['benefice_from_baseline'].quantile(0.5),
            df['benefice_from_baseline'].mean(),
            df['benefice_from_baseline'].quantile(0.975),
            log_transform(df['benefice_from_baseline'].quantile(0.025)),
            log_transform(df['benefice_from_baseline'].quantile(0.5)),
            log_transform(df['benefice_from_baseline'].mean()),
            log_transform(df['benefice_from_baseline'].quantile(0.975)),
            df['cumulated_payoff'].quantile(0.025),
            df['cumulated_payoff'].quantile(0.5),
            df['cumulated_payoff'].mean(),
            df['cumulated_payoff'].quantile(0.975),
            df['baseline_payoff'].quantile(0.025),
            df['baseline_payoff'].quantile(0.5),
            df['baseline_payoff'].mean(),
            df['baseline_payoff'].quantile(0.975),
            nbr_negative_ben,
            nbr_negative_ben / nombre_sim
        ]],
            columns = summary_columns)],
                                 ignore_index=False)

        # Compiling data
        data = pd.concat([data, df[['country_code', 'benefice_from_baseline', 'cumulated_payoff', 'sim_id']]], axis = 0)

        # Individual plot
        if plot_individual_results:
            fig, axes = plt.subplots(1, 2)
            fig.suptitle(f"Distributions of the {experiment_strategy} strategy payoffs and relative gains/losses for {countries_names_dict[country_code]} ({country_code})")
            sns.kdeplot(df['cumulated_payoff'], color = 'blue', shade = True, ax = axes[0], label = f'{experiment_strategy}')
            sns.kdeplot(df['baseline_payoff'], color = 'purple', shade = True, ax = axes[0], label = 'baseline')

            tick = mtick.StrMethodFormatter(fmt)
            axes[0].set_title('Distribution of payoffs')
            axes[0].set(xlabel = None)
            # axes[0].ticklabel_format(style='plain', axis='x', useOffset=False)
            axes[0].xaxis.set_major_formatter(tick)
            axes[0].tick_params(axis='x', labelrotation = 25)
            axes[0].legend(loc ='upper left')
            g = sns.kdeplot(df['benefice_from_baseline'], color = 'blue', shade = True, ax = axes[1])

            # create the kde model
            kde = gaussian_kde(df['benefice_from_baseline'])
            # get the min and max of the x-axis
            xmin, xmax = g.get_xlim()
            # create points between the min and max
            x = np.linspace(xmin, xmax, 1000)
            # calculate the y values from the model
            kde_y = kde(x)
            # select x values below 0
            x0 = x[x < 0]
            # get the len, which will be used for slicing the other arrays
            x0_len = len(x0)
            # slice the arrays
            y0 = kde_y[:x0_len]
            x1 = x[x0_len:]
            y1 = kde_y[x0_len:]
            # calculate the area under the curves
            # area0 = np.round(simps(y0, x0, dx=1) * 100, 0)
            # area1 = np.round(simps(y1, x1, dx=1) * 100, 0)
            # fill the areas
            g.fill_between(x=x0, y1=y0, color='r', alpha=.3)
            g.fill_between(x=x1, y1=y1, color='g', alpha=.3)

            axes[1].set_title('Distribution of gains/losses compared to the baseline')
            axes[1].set(xlabel = None)
            # axes[1].ticklabel_format(style='plain', axis='x', useOffset=False)
            axes[1].xaxis.set_major_formatter(tick)
            axes[1].tick_params(axis='x', labelrotation = 25)

            plt.show()
            fig.savefig(f'../img/{experiment_strategy}/payoff_distribution_{country_code}_{experiment_strategy}.pdf',
                        dpi='figure',
                        format='pdf',
                        metadata=None,
                        bbox_inches=None,
                        pad_inches=0.1,
                        facecolor='auto',
                        edgecolor='auto',
                        backend=None)

    logging.info("Loop over countries finished")

    # Save summary data
    if save_summary :
        logging.info("Saving summary data")
        summary_data.to_csv(f'../results/summary_data_{experiment_strategy}.csv', encoding ='UTF-8', sep = ';', decimal='.')

    # Save compiled data
    if save_compiled_data:
        logging.info("Saving compiled data")
        data.to_csv(f'../results/compiled_data_{experiment_strategy}.csv', encoding ='UTF-8', sep = ';', decimal='.')

    logging.info("Finished")

def plot_comparing_payoffs(experiment_strategy:str = 'one_vac', folder_name:str='../results'):
    """
    Plot the payoffs of the different strategies for all countries corresponding to the figures 3 (all_vac) and 4 (one_vac) in the appendix.
    Needed data : summary_data_{experiment_strategy}.csv
    :param experiment_strategy: the strategy to plot the payoffs for (Possible values
    "one_vac", "all_vac")
    :return: None
    """
    logging.info("Starting plotting of payoffs")

    # Read data
    logging.debug("Reading data : starting")
    summary_data = pd.read_csv(f'{folder_name}/summary_data_{experiment_strategy}.csv', encoding ='UTF-8', sep = ';', decimal='.')
    logging.debug("Reading data : finished")

    summary_data.sort_values(by = 'gain_loss_to_baseline_mean', inplace = True)

    logging.debug("Starting plotting")
    fig, axes = plt.subplots(2,1)

    x_pos = np.arange(0,48,1)

    fig.suptitle(
            f"Countries {experiment_strategy} strategy relative gains/losses with 2.5% and 97.5% percentiles (linear and log scales)")

    colors_list = ['#006400', '#FF0000']
    custom_palette = sns.color_palette(colors_list, 2)
    summary_data['color'] = summary_data.apply(lambda x : 0 if x['gain_loss_to_baseline_mean'] > 0 else 1, axis = 1)
    summary_data['color_2'] = summary_data.apply(lambda x : 0 if x['gain_loss_to_baseline_mean_log'] > 0 else 1, axis = 1)

    sns.scatterplot(data = summary_data,
                    x = 'country_code',
                    y ='gain_loss_to_baseline_mean',
                    marker = 'x',
                    hue = 'color',
                    ax = axes[0],
                    palette = custom_palette,
                    legend = False,
                    linewidth = 2.5)
        
    axes[0].vlines(x_pos,
                       summary_data['gain_loss_to_baseline_0025'].values,
                       summary_data['gain_loss_to_baseline_0975'].values,
                       colors = 'k',
                       linestyles = 'solid',
                        linewidth = 1)

    axes[0].scatter(x_pos, summary_data['gain_loss_to_baseline_0025'].values, color = 'k', marker = '_')
    axes[0].scatter(x_pos, summary_data['gain_loss_to_baseline_0975'].values, color = 'k', marker = '_')

    tick = mtick.StrMethodFormatter(fmt)
    axes[0].yaxis.set_major_formatter(tick)
    axes[0].set_xticks([])
    axes[0].set(ylabel='Relative gains/losses (in $)', xlabel = None)
    axes[0].set_title(label = 'a) Relative gains/losses on a linear scale.', loc = 'left')
    axes[0].hlines(0, -1, 49, colors = 'k', linestyles = 'solid', linewidth = 0.5, alpha = 0.5)
    axes[0].grid(visible = True, which = 'both', axis = 'y',
                 color = 'k', linestyle = '--', linewidth = 0.5, alpha = 0.5)

    sns.scatterplot(data = summary_data,
                    x = 'country_code',
                    y ='gain_loss_to_baseline_mean_log',
                    ax = axes[1],
                    marker='x',
                    hue='color_2',
                    palette =  custom_palette,
                    legend = False,
                    linewidth = 2.5)

    axes[1].vlines(x_pos,
                       summary_data['gain_loss_to_baseline_0025_log'].values,
                       summary_data['gain_loss_to_baseline_0975_log'].values,
                       colors = 'k',
                       linestyles = 'solid',
                       linewidth = 1)
    axes[1].hlines(0, -1, 49, colors = 'k', linestyles = 'solid', linewidth = 0.5, alpha = 0.5)
    axes[1].scatter(x_pos, summary_data['gain_loss_to_baseline_0025_log'].values, color = 'k', marker = '_')
    axes[1].scatter(x_pos, summary_data['gain_loss_to_baseline_0975_log'].values, color = 'k', marker = '_')
    axes[1].set(ylabel= 'Special log transformation of gains/losses (cf. methodology)', xlabel = 'Country ISO-3 code' )
    axes[1].set_title(label = 'b) Relative gains/losses on a special logarithmic scale.', loc = 'left')
    axes[1].grid(visible = True, which = 'major', axis = 'y',
                 color = 'k', linestyle = '--', linewidth = 0.5, alpha = 0.5)

    plt.show()

    fig.savefig('../img/figure_{}_appendix.pdf'.format(3 if experiment_strategy=='all_vac' else 4),
                    dpi='figure',
                    format='pdf',
                    metadata=None,
                    bbox_inches=None,
                    pad_inches=0.1,
                    facecolor='auto',
                    edgecolor='auto',
                    backend=None)

    # fig.savefig('../img/figure_{}_appendix.png'.format(3 if experiment_strategy=='all_vac' else 4),
    #                 dpi='figure',
    #                 format=None,
    #                 metadata=None,
    #                 bbox_inches=None,
    #                 pad_inches=0.1,
    #                 facecolor='auto',
    #                 edgecolor='auto',
    #                 backend=None)

    logging.info("Finished")

def plot_and_save_total_payoff(folder_name:str='../results'):
    """
    Plot the total payoffs distributions.
    Needed data : compiled_data_all_vac.csv
    :return:
    """
    logging.info("Starting plotting of total payoffs")

    # Read the data
    logging.debug("Reading data : starting")
    df = pd.read_csv(f'{folder_name}/compiled_data_all_vac.csv', encoding ='UTF-8', sep = ';', decimal='.')
    logging.debug("Reading data : finished")

    # print(df.head(10))
    groupby_sim = df.groupby('sim_id')[['benefice_from_baseline', 'cumulated_payoff']].sum()
    groupby_sim['baseline_payoff'] = groupby_sim['cumulated_payoff'] - groupby_sim['benefice_from_baseline']
    # print(groupby_sim)
    print('Mean value for the gains/losses compared to the baseline : ${:,.0f}'.format(groupby_sim['benefice_from_baseline'].mean()))
    print('Percentile interval [2.5%, 97.5%] for the gains/losses compared to the baseline: [${:,.0f}, ${:,.0f}]'.format(groupby_sim['benefice_from_baseline'].quantile(q = 0.025),
                                                                                                           groupby_sim['benefice_from_baseline'].quantile(q = 0.975)))

    logging.debug("Starting plotting")
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(
        f"Distributions of total payoffs and relative gains/losses")
    sns.kdeplot(groupby_sim['cumulated_payoff'], color='blue', shade=True, ax=axes[0], label=f'all vaccination')
    sns.kdeplot(groupby_sim['baseline_payoff'], color='purple', shade=True, ax=axes[0], label='baseline')

    tick = mtick.StrMethodFormatter(fmt)
    axes[0].set_title('Distribution of payoffs')
    axes[0].set(xlabel=None)
    # axes[0].ticklabel_format(style='plain', axis='x', useOffset=False)
    axes[0].xaxis.set_major_formatter(tick)
    axes[0].tick_params(axis='x', labelrotation=25)
    axes[0].legend(loc='upper left')
    g = sns.kdeplot(groupby_sim['benefice_from_baseline'], color='blue', shade=True, ax=axes[1])

    # create the kde model
    kde = gaussian_kde(groupby_sim['benefice_from_baseline'])
    # get the min and max of the x-axis
    xmin, xmax = g.get_xlim()
    # create points between the min and max
    x = np.linspace(xmin, xmax, 1000)
    # calculate the y values from the model
    kde_y = kde(x)
    # select x values below 0
    x0 = x[x < 0]
    # get the len, which will be used for slicing the other arrays
    x0_len = len(x0)
    # slice the arrays
    y0 = kde_y[:x0_len]
    x1 = x[x0_len:]
    y1 = kde_y[x0_len:]
    # calculate the area under the curves
    # area0 = np.round(simps(y0, x0, dx=1) * 100, 0)
    # area1 = np.round(simps(y1, x1, dx=1) * 100, 0)
    # fill the areas
    g.fill_between(x=x0, y1=y0, color='r', alpha=.3)
    g.fill_between(x=x1, y1=y1, color='g', alpha=.3)

    axes[1].set_title('Distribution of gains/losses compared to the baseline')
    axes[1].set(xlabel=None)
    # axes[1].ticklabel_format(style='plain', axis='x', useOffset=False)
    axes[1].xaxis.set_major_formatter(tick)
    axes[1].tick_params(axis='x', labelrotation=25)

    plt.show()
    fig.savefig(f'../img/payoff_distribution_all_vac_all_countries.pdf',
                dpi='figure',
                format='pdf',
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto',
                backend=None)

    logging.info("Finished")

def compile_and_save_dog_population_data(folder_name:str='../results'):
    """
    Compile the resulting data from the Monte Carlo simulations for rabid dog population, exposed humans and clinical cases.
    Needed data from the sensitivity analysis: simulation_dog_population_results_{country}.csv

    :return:
    """

    logging.info("Starting compiling data from Monte-Carlo simulations for rabid dog population, exposed humans and clinical cases")

    df = pd.DataFrame(columns = ['country_code', 'rabid_dogs_population', 'exposed_humans', 'clinical_cases', 'total_dog_population','sim_id'])

    logging.info("Starting looping over countries")
    # Loop over the countries for compiling
    for country in countries_codes_list:
        logging.debug(f"Processing for the country code: {country}")

        inter = pd.read_csv(f'{folder_name}/dog_population/simulation_dog_population_results_{country}.csv', encoding = 'UTF-8', sep = ';', decimal = '.', index_col=0)
        inter['sim_id'] = range(200000)
        df = pd.concat([df, inter], axis = 0,ignore_index = True)

    logging.info("Finished looping over countries")

    # Save the compiled data
    logging.info("Saving compiled data")
    df.to_csv('../results/compiled_data_rabid_dog_population.csv', encoding='UTF-8', sep =';', decimal = '.')
    logging.info("Saving compiled data : finished")

    # Print some information
    groupby = df.groupby('sim_id')[['rabid_dogs_population', 'exposed_humans', 'clinical_cases']].sum()
    print('Mean rabid dog population at baseline: {:,.0f}'.format(groupby['rabid_dogs_population'].mean()))
    print('Percentile interval [2.5%, 97.5%] for rabid dog population at baseline: [{:,.0f}, {:,.0f}]'.format(groupby['rabid_dogs_population'].quantile(q = 0.025),
                                                                                                           groupby['rabid_dogs_population'].quantile(q = 0.975)))

    print('Mean value of exposed humans at the baseline: {:,.0f}'.format(groupby['exposed_humans'].mean()))
    print('Percentile interval [2.5%, 97.5%] for the number of exposed humans at the baseline: [{:,.0f}, {:,.0f}]'.format(groupby['exposed_humans'].quantile(q = 0.025),
                                                                                                           groupby['exposed_humans'].quantile(q = 0.975)))

    print('Mean value of clinical cases at the baseline : {:,.0f}'.format(groupby['clinical_cases'].mean()))
    print('Percentile interval [2.5%, 97.5%] for the number of clinical cases at the baseline: [{:,.0f}, {:,.0f}]'.format(groupby['clinical_cases'].quantile(q = 0.025),
                                                                                                           groupby['clinical_cases'].quantile(q = 0.975)))

    logging.debug("Plotting the distributions")
    # Plot
    fig, axes = plt.subplots(1, 3)
    fig.suptitle(
        f"Distributions of dog populations, exposed humans and clinical cases at the baseline")
    sns.kdeplot(groupby['rabid_dogs_population'], color='blue', shade=True, ax=axes[0], label=f'all vaccination')
    sns.kdeplot(groupby['exposed_humans'], color='purple', shade=True, ax=axes[1], label='baseline')
    sns.kdeplot(groupby['clinical_cases'], color='red', shade=True, ax=axes[2], label='baseline')

    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)

    axes[0].set_title('Distribution of the rabid dog population')
    axes[0].xaxis.set_major_formatter(tick)
    axes[0].tick_params(axis='x', labelrotation=25)
    axes[0].set(xlabel=None)

    axes[1].set_title('Distribution of the exposed humans')
    axes[1].xaxis.set_major_formatter(tick)
    axes[1].tick_params(axis='x', labelrotation=25)
    axes[1].set(xlabel=None)

    axes[2].set_title('Distribution of the clinical cases')
    axes[2].xaxis.set_major_formatter(tick)
    axes[2].tick_params(axis='x', labelrotation=25)
    axes[2].set(xlabel=None)


    plt.show()
    fig.savefig(f'../img/rabid_dog_population_distribution_baseline.pdf',
                dpi='figure',
                format='pdf',
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto',
                backend=None)

    logging.info("Finished")


def coalition_analysis(folder_name:str='../results'):
    """
    Coalition analysis with confidence interval. Payoffs comparison between different strategies possibilities.

    Needed data :

    - compiled_data_all_vac.csv
    - compiled_data_one_vac.csv
    :return:
    """

    logging.info("Starting coalition analysis")

    def coalition_group(row) :
        if row in coalition_pep:
            return 'C_PEP'
        else:
            return 'C_VAC'

    # Read data
    logging.debug("Reading data: 1/2")
    df = pd.read_csv(f'{folder_name}/compiled_data_all_vac.csv', encoding ='UTF-8', sep = ';', decimal='.')

    df['coalition'] = df['country_code'].apply(lambda row: coalition_group(row))

    # Print some information
    groupby_coalition_sim = df.groupby(by = ['coalition', 'sim_id'], as_index=False)['benefice_from_baseline'].sum()
    print('Mean value for the gains/losses (ALL VAC) compared to the baseline for coalition C_PEP: ${:,.0f}'.format(
        groupby_coalition_sim.loc[groupby_coalition_sim['coalition']=='C_PEP','benefice_from_baseline'].mean()))
    print(
        'Percentile interval [2.5%, 97.5%] for the gains/losses compared to the baseline coalition C_PEP: [${:,.0f}, ${:,.0f}]'.format(
            groupby_coalition_sim.loc[groupby_coalition_sim['coalition']=='C_PEP','benefice_from_baseline'].quantile(q=0.025),
            groupby_coalition_sim.loc[groupby_coalition_sim['coalition']=='C_PEP','benefice_from_baseline'].quantile(q=0.975)))

    print('Mean value for the gains/losses compared (ALL VAC) to the baseline for coalition C_VAC: ${:,.0f}'.format(
        groupby_coalition_sim.loc[groupby_coalition_sim['coalition']=='C_VAC','benefice_from_baseline'].mean()))
    print(
        'Percentile interval [2.5%, 97.5%] for the gains/losses compared to the baseline coalition C_VAC: [${:,.0f}, ${:,.0f}]'.format(
            groupby_coalition_sim.loc[groupby_coalition_sim['coalition']=='C_VAC','benefice_from_baseline'].quantile(q=0.025),
            groupby_coalition_sim.loc[groupby_coalition_sim['coalition']=='C_VAC','benefice_from_baseline'].quantile(q=0.975)))

    # Read data
    logging.debug("Reading data: 2/2")
    df = pd.read_csv(f'{folder_name}/compiled_data_one_vac.csv', encoding ='UTF-8', sep = ';', decimal='.')
    logging.debug("Reading data: finished")

    df['coalition'] = df['country_code'].apply(lambda row: coalition_group(row))

    # Print some information
    groupby_coalition_sim = df.groupby(by = ['coalition', 'sim_id'], as_index=False)['benefice_from_baseline'].sum()
    print('Mean value for the gains/losses (ONE VAC) compared to the baseline for coalition C_PEP: ${:,.0f}'.format(
        groupby_coalition_sim.loc[groupby_coalition_sim['coalition']=='C_PEP','benefice_from_baseline'].mean()))
    print(
        'Percentile interval [2.5%, 97.5%] for the gains/losses compared to the baseline coalition C_PEP: [${:,.0f}, ${:,.0f}]'.format(
            groupby_coalition_sim.loc[groupby_coalition_sim['coalition']=='C_PEP','benefice_from_baseline'].quantile(q=0.025),
            groupby_coalition_sim.loc[groupby_coalition_sim['coalition']=='C_PEP','benefice_from_baseline'].quantile(q=0.975)))

    print('Mean value for the gains/losses compared (ONE VAC) to the baseline for coalition C_VAC: ${:,.0f}'.format(
        groupby_coalition_sim.loc[groupby_coalition_sim['coalition']=='C_VAC','benefice_from_baseline'].mean()))
    print(
        'Percentile interval [2.5%, 97.5%] for the gains/losses compared to the baseline coalition C_VAC: [${:,.0f}, ${:,.0f}]'.format(
            groupby_coalition_sim.loc[groupby_coalition_sim['coalition']=='C_VAC','benefice_from_baseline'].quantile(q=0.025),
            groupby_coalition_sim.loc[groupby_coalition_sim['coalition']=='C_VAC','benefice_from_baseline'].quantile(q=0.975)))

    logging.info("Finished")

def visualize_gains(folder_name:str='../results'):
    """
    Visualize gains on the map of Africa.

    Needed data :

    - Appendix 4 - Data for simulations.xlsx
    - summary_data_all_vac.csv
    :return:
    """
    logging.info("Starting visualization of gains")

    # Load shapefile of African countries
    logging.debug("Reading data: shapefile")
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    africa = world.query('continent == "Africa"')

    # Load GDP data
    logging.debug("Reading data: 1/2")
    gdp_data = pd.read_excel("../data/Appendix 4 - Data for simulations.xlsx",
                                     sheet_name = "Pop_Data",
                                     header = 0)

    gdp_data['nash_strategy'] = np.array([
            0,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1
        ])

    logging.debug("Reading data: 2/2")
    gains_data = pd.read_csv(f'{folder_name}/summary_data_all_vac.csv', encoding='UTF-8', sep = ';', decimal= ',')
    logging.debug("Reading data: finished")


    # Merge the data
    # africa_map = pd.merge(africa, gdp_data, on='Country')
    africa_map = pd.merge(africa, gains_data, left_on='iso_a3', right_on = 'country_code')

    # Correcting Marocco shape
    x_mar = [-2.16991370279862, -1.79298580566171, -1.73345455566146, -1.38804928222259, -1.1245511539663,
             -1.30789913573787, -2.61660478352956, -3.06898027181264, -3.64749793132014, -3.69044104655472,
             -4.85964616537447, -5.24212927898278, -6.06063229005377, -7.05922766766195, -8.67411617678297,
             -8.6655895654548, -8.68439978680905, -8.6872936670174, -11.9694189111711, -11.9372244938533,
             -12.8742215641695,
             -13.1187544417747, -12.9291019352635, -16.8451936507739, -17.002961798561, -17.0204284326757,
             -16.9732478499932, -16.5891369287676, -16.2619217594956, -16.3264139469959, -15.982610642958,
             -15.4260037907421, -15.0893318343607, -14.8246451481616, -14.8009256657397, -14.4399399479648,
             -13.7738048975064, -13.1399417790143, -13.1216133699147, -12.6188366357831, -11.6889192366907,
             -10.9009569971044, -10.3995922510086, -9.56481116376568, -9.81471839032917, -9.43479326011936,
             -9.30069291832188, -8.65747636558501, -7.65417843263821, -6.91254411460141, -6.24434200685141,
             -5.92999426921989, -5.19386349122203, -4.59100623210514, -3.64005652507006, -2.60430579264408,
             -2.16991370279862]
    y_mar = [35.1683963079166, 34.5279186060913, 33.9197128362321, 32.8640150009413, 32.6515215113571, 32.2628889023061,
             32.0943462183861, 31.7244979924732, 31.6372940129806, 30.8969516057511, 30.5011876490438, 30.0004430201355,
             29.7316997340016, 29.5792284205246, 28.8412889673965, 27.6564258895923, 27.395744126896, 25.8810562199889,
             25.9333527694682, 23.3745942245361, 23.2848322616451, 22.7712202010962, 21.3270706242675, 21.3333234725748,
             21.4207341577965, 21.4223102889815, 21.8857445337749, 22.15823436125, 22.6793395044812, 23.0177684595609,
             23.723358466074, 24.359133612561, 24.520260728447, 25.1035326197253, 25.6362649602223, 26.2544184432976,
             26.6188923202523, 27.6401478134205, 27.6541476717198, 28.0381855331486, 28.1486439071725, 28.8321422388809,
             29.0985859237778, 29.9335737167498, 31.177735500609, 32.0380964218364, 32.5646792668906, 33.2402452662424,
             33.6970649277025, 34.1104763860374, 35.1458653834375, 35.7599881047939, 35.7551821965908, 35.3307119817455,
             35.399855048152, 35.1790933294011, 35.1683963079166]

    africa_map.loc[africa_map['iso_a3'] == 'MAR', 'geometry'] = Polygon(zip(x_mar, y_mar))

    #africa_map.to_csv("map.csv", encoding='UTF-8', sep = ';')
    # print(africa_map)

    africa_map.drop(columns=['country_code'], inplace = True)
    africa_map = pd.merge(africa_map, gdp_data[['country_code', 'gdp_2024', 'nash_strategy']], left_on='iso_a3', right_on = 'country_code')
    africa_map.drop(columns=['country_code'], inplace = True)
    logging.debug(africa_map)
    africa_map['gains_by_gdp'] = africa_map['gain_loss_to_baseline_mean'].astype(float)/africa_map['gdp_2024']
    africa_map['gain_loss_to_baseline_mean'] = africa_map['gain_loss_to_baseline_mean'].astype(float)
    # Set up the plot
    logging.info("Plotting map: 1/2")
    fig, ax = plt.subplots(figsize=(10, 8))
    # Add Labels

    africa_map['coords'] = africa_map['geometry'].apply(lambda x: x.representative_point().coords[:])
    africa_map['coords'] = [coords[0] for coords in africa_map['coords']]

    ax.set_title('Gains in USD from the cooperation for mass dog vaccination')
    ax.axis('off')
    # set the range for the choropleth values
    vmin, vmax = 0, 1500000000

    # Create colorbar legend
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin, vmax=vmax))

    # empty array for the data range
    sm.set_array([])  # or alternatively sm._A = []. Not sure why this step is necessary, but many recommends it
    cb = fig.colorbar(sm,fraction=0.036, pad=0.04)
    cb.ax.set_yticklabels(['${:,.0f}'.format(i) for i in cb.get_ticks()])  # set ticks of your format

   # Plot the choropleth
    africa_map.plot(column='gain_loss_to_baseline_mean',
                    cmap='Reds',
                    ax=ax,
                    legend=False,
                    edgecolor='k'
                    )
    # africa_map[africa_map['nash_strategy'] == 0].plot(ax=ax, facecolor='none', edgecolor='0.8', hatch='///')
    africa_map[africa_map['nash_strategy'] == 1].plot(ax=ax,
                                                      facecolor='none',
                                                      edgecolor='k',
                                                      linewidth = 0.3,
                                                      alpha = 0.5,
                                                      hatch='xxx')


    # # Add a custom legend
    legend_elements = [
        # Line2D([0], [0], color='white', lw=0, label='Vaccination as dominant strategy',
        #        markerfacecolor='white',
        #        marker='', markersize=10, markeredgecolor='black', markeredgewidth=0.5),
        Patch(facecolor='w', edgecolor='black', label='Vaccination as dominant strategy', hatch='xxx'),
        Patch(facecolor='w', edgecolor='black', label='No dominant strategy'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',  frameon=False) #bbox_to_anchor=(0,0),

    # Show the plot
    plt.show()
    fig.savefig(f'../img/figure_2_manuscript.pdf',
                dpi='figure',
                format='pdf',
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto',
                backend=None)

    logging.info("Plotting map: 2/2")
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_title('Gains in % of the GDP from the cooperation for mass dog vaccination')
    ax.axis('off')
    # set the range for the choropleth values
    vmin, vmax = 0, 0.02688

    # Create colorbar legend
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))

    # empty array for the data range
    sm.set_array([])  # or alternatively sm._A = []. Not sure why this step is necessary, but many recommends it
    cb = fig.colorbar(sm,fraction=0.036, pad=0.04)
    cb.ax.set_yticklabels(['{:,.1%}'.format(i) for i in cb.get_ticks()])  # set ticks of your format

   # Plot the choropleth
    africa_map.plot(column='gains_by_gdp',
                    cmap='Blues',
                    ax=ax,
                    legend=False,
                    edgecolor='k',
                    )
    plt.show()
    fig.savefig(f'../img/figure_3_manuscript.pdf',
                dpi='figure',
                format='pdf',
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto',
                backend=None)

    logging.info("Finished")

def exposure_plot(folder_name = '../results'):
    """
    Plot the number of exposed humans over the years under different scenarios.

    Needed data:

    -   exposed_humans_{filename}.csv (filename in ['all_pep', 'all_vac', 'nash', 'n24', 'one_pep'])

    :return:
    """

    logging.info("Starting exposure plot")

    files_names = ['all_pep', 'all_vac', 'nash', 'n24', 'one_pep']
    res = pd.DataFrame(data = {"year" :  range(2024, 2055)})

    logging.info("Starting loop for reading data")
    # Read data
    for file_name in files_names:

        logging.debug("Reading data for " + file_name)
        df = pd.read_csv(f"{folder_name}/results_by_year/exposed_humans_{file_name}.csv",
                         encoding = 'UTF-8',
                         sep = ';',
                         decimal = '.',
                         index_col = 0)

        df[file_name.lower()] = np.sum(df[countries_codes_list], axis=1)

        if file_name in ['n24', 'one_pep']:
            print(df)
            df = df.groupby(by = 'year', as_index=False)[file_name].mean()
            # print(df)

        res = pd.concat([res, df[[file_name.lower()]]], axis = 1)

    logging.info("Finished loop for reading data")

    initial_pop =  398529 / (1+0.012694042)

    res = pd.concat([
        pd.DataFrame(data =  [[2023, initial_pop, initial_pop, initial_pop, initial_pop, initial_pop]],
                     columns = ['year', 'all_pep', 'all_vac', 'nash', 'n24', 'one_pep']),
        res
    ], axis=0, ignore_index=True)

    logging.debug(res)

    # Plot
    logging.info("Starting plot")
    fig, ax = plt.subplots(1,1)
    plt.title("Total Number of Humans Exposed to Rabid Dogs per Year for all 48 countries depending on the strategy profile")
    plt.plot(res['year'], res['all_vac'], color = 'g', linestyle = '-', marker = 'x', label = 'Vaccination campaign in all countries')
    plt.plot(res['year'], res['all_pep'], color = 'r', linestyle = '-', marker = 'x', label = 'Baseline (No vaccination)')
    plt.plot(res['year'], res['nash'], color = 'k', linestyle = '--', marker = 'x', label = 'Nash equilibrium (vaccination where it is a dominant strategy)')
    plt.plot(res['year'], res['one_pep'], color = 'c', linestyle = '--', marker = '.', label = 'Average for strategy profile with one defecting country')
    plt.plot(res['year'], res['n24'], color = 'b', linestyle = '--', marker = '.', label = 'Average for strategy profile with half vaccinated')
    plt.legend()
    plt.grid()
    plt.ylabel('Number of Humans Exposed')
    plt.xlabel('Year')
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    plt.show()

    fig.savefig(f'../img/figure_1_manuscript.pdf',
                dpi='figure',
                format='pdf',
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto',
                backend=None)
    logging.info("Finished")

def break_even_calculation(hce = False, folder_name = '../results'):
    """
    Plot breakeven years with and without consideration for HCE.

    Needed data (generated by the script strategy_analysis.py):

    -   payoff_total_ALL_PEP.csv
    -   payoff_total_ALL_VAC.csv
    -   payoff_without_hce_ALL_PEP.csv
    -   payoff_without_hce_ALL_VAC.csv

    :param hce: Include HCE or not.
    :return:
    """

    logging.info("Starting break even calculation")

    if hce:
        file_name_prefix = 'payoff_total'
        file_name_figure = 'figure_8_appendix'
        title = 'Break-even point per country for the costs of the vaccination campaign and' \
                ' PEP use (including the Human Capital Effect (HCE))'
    else :
        file_name_prefix = 'payoff_without_hce'
        file_name_figure = 'figure_7_appendix'
        title = 'Break-even point per country for the costs of the vaccination campaign and' \
                ' PEP use (without the Human Capital Effect (HCE))'
    # Read data
    logging.debug("Reading data: 1/2")
    df_pep = pd.read_csv(f"{folder_name}/results_by_year/{file_name_prefix}_all_pep.csv",
                         encoding='UTF-8',
                         sep=';',
                         decimal='.',
                         index_col=0)

    logging.debug("Reading data: 2/2")
    df_vac = pd.read_csv(f"{folder_name}/results_by_year/{file_name_prefix}_all_vac.csv",
                         encoding='UTF-8',
                         sep=';',
                         decimal='.',
                         index_col=0)

    # Calculate the cumulative payoffs
    df_pep = df_pep.cumsum()
    df_vac = df_vac.cumsum()

    # Calculate the difference
    res = df_vac - df_pep

    try:
        res.drop(columns = ['Year'], inplace =True)
    except: pass

    # Replace values by sign of the difference
    res[res>0] = 1
    res[res<0] = 0

    res['year'] = range(2024, 2055)

    # Plot
    logging.debug("Starting plot")
    fig, ax = plt.subplots()

    plt.suptitle(title)
    im = plt.imshow(res[countries_codes_list].T, cmap='RdYlGn')

    # Create colorbar
    qrates = ['Cost vaccination < Cost PEP', 'Cost vaccination > Cost PEP']
    norm = BoundaryNorm([0., 0.5, 1.], 2, clip = True)
    fmt = FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[0., 1.], format=fmt)
    cbar.ax.set_ylabel('', rotation=-90, va="bottom")

    ax.set_yticks(np.arange(len(countries_codes_list)), labels=countries_codes_list)
    ax.set_xticks(np.arange(len(res['year'])), labels=res['year'])
    plt.setp(ax.get_xticklabels(), rotation=-45, horizontalalignment = 'center',
             verticalalignment='top', rotation_mode="default")

    # Minor ticks
    plt.hlines(y=np.arange(0, 48) + 0.5, xmin=np.full(48, 0) - 0.5,
               xmax=np.full(48, 31) - 0.5, color="w", linewidths = 1)
    plt.vlines(x=np.arange(0, 31) + 0.5, ymin=np.full(31, 0) - 0.5,
               ymax=np.full(31, 48) - 0.5, color="w", linewidths = 1)

    plt.show()
    fig.savefig(f'../img/{file_name_figure}.png',
                dpi='figure',
                format=None,
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto',
                backend=None)

    fig.savefig(f'../img/{file_name_figure}.pdf',
                dpi='figure',
                format='pdf',
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto',
                backend=None)

    logging.info("Finished")

    return im

def cost_analysis_plot(hce = True, folder_name = '../results'):
    """
    Plot the cost analysis over the years.

    Needed data (generated by the script strategy_analysis.py):
    - payoff_total_ALL_PEP.csv
    - payoff_total_ALL_VAC.csv
    - payoff_without_hce_ALL_PEP.csv
    - payoff_without_hce_ALL_VAC.csv
    - payoff_total_NASH.csv
    - payoff_without_hce_NASH.csv
    - payoff_total_n24.csv
    - payoff_without_hce_n24.csv
    - payoff_total_one_pep.csv
    - payoff_without_hce_one_pep.csv

    :param hce:
    :return:
    """

    logging.info("Starting cost analysis plot")

    if hce:
        file_name_prefix = 'payoff_total'
    else:
        file_name_prefix = 'payoff_without_hce'


    files_names = ['all_pep', 'all_vac', 'nash', 'n24', 'one_pep']
    res = pd.DataFrame(data={"year": range(2024, 2055)})

    logging.info("Starting looping over the files for reading data")

    # Read the data
    for file_name in files_names:

        logging.debug("Reading data: " + file_name)

        df = pd.read_csv(f"{folder_name}/results_by_year/{file_name_prefix}_{file_name}.csv",
                         encoding='UTF-8',
                         sep=';',
                         decimal='.',
                         index_col=0)

        df[file_name.lower()] = np.sum(df[countries_codes_list], axis=1)

        if file_name in ['n24', 'one_pep']:
            df = df.groupby('year', as_index=False)[file_name].mean()
            print(df)

        res = pd.concat([res, df[[file_name.lower()]]], axis=1)

    res = pd.concat([
        pd.DataFrame(data =  [[2023, 0, 0, 0, 0, 0]],
                     columns = ['year', 'all_pep', 'all_vac', 'nash', 'n24', 'one_pep']),
        res
    ], axis=0, ignore_index=True)

    logging.info("Finished looping over the files for reading data")

    # Calculate the cumulative payoffs
    res_cumsum = res.copy()
    res_cumsum[['all_pep', 'all_vac', 'nash', 'n24', 'one_pep']] = res[['all_pep', 'all_vac', 'nash', 'n24', 'one_pep']].cumsum()

    # Plot
    logging.debug("Starting plot")
    fig, ax = plt.subplots(1,1)
    plt.title("Cost {} per Year for all 48 countries depending on the strategy profile".format('(with HCE)' if hce else '(without HCE)'))
    plt.plot(res_cumsum['year'], res_cumsum['all_vac'], color = 'g', linestyle = '-', marker = 'x', label = 'Vaccination campaign in all countries')
    plt.plot(res_cumsum['year'], res_cumsum['all_pep'], color = 'r', linestyle = '-', marker = 'x', label = 'Baseline (No vaccination)')
    plt.plot(res_cumsum['year'], res_cumsum['nash'], color = 'k', linestyle = '--', marker = 'x', label = 'Nash equilibrium (vaccination where it is a dominant strategy)')
    plt.plot(res_cumsum['year'], res_cumsum['one_pep'], color = 'c', linestyle = '--', marker = '.', label = 'Average for strategy profile with one defecting country')
    plt.plot(res_cumsum['year'], res_cumsum['n24'], color = 'b', linestyle = '--', marker = '.', label = 'Average for strategy profile with half vaccinated')
    plt.legend()
    plt.grid()
    plt.ylabel('')
    plt.xlabel('Year')
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    plt.show()

    fig.savefig('../img/figure_{}_appendix.pdf'.format(6 if hce else 5),
                dpi='figure',
                format='pdf',
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto',
                backend=None)

    logging.info("Finished")


def estimate_confidence_interval_lives_lost(country_code, folder_name='../results'):
    """
    Estimate the confidence interval for the number of lives lost due to rabies at the baseline.
    The data for the dog population is generated by the script monte_carlo_baseline_populations.py.

    Needed data:
    - Appendix 4 - Data for simulations.xlsx
    - simulation_dog_population_results_{country_code}.csv

    :param country_code: The country code of the country we want to estimate the confidence interval.
    :return:
    """
    logging.info("Starting confidence interval estimation for the number of lives lost for {}".format(country_code))
    # print("==================== Running Monte Carlo for the number of lives lost for {} ======================".format(country_code))

    # Read the data for the probability of receiving PEP
    logging.debug("Reading data for the probability of receiving PEP: 1/2")
    df = pd.read_excel("../data/Appendix 4 - Data for simulations.xlsx", #./Alvar/Dog Population Knobel 2005 including PopGrowth.xlsx
                       sheet_name="Pop_Data",
                       header=0)
    df = df[df['country_code']==country_code]

    # Read the data from Monte Carlo simulations
    logging.debug("Reading data for the dog population: 2/2")
    df_res = pd.read_csv(f'{folder_name}/dog_population/simulation_dog_population_results_{country_code}.csv', encoding='UTF-8', sep = ';', decimal='.')

    # Build the empirical distribution using Kernel Density Estimator
    sample = np.array(df_res['clinical_cases'].values)
    sample = np.expand_dims(sample, axis=1)
    ks = ot.KernelSmoothing()
    fittedDist = ks.build(sample)

    # Define the distribution for the probability of receiving PEP
    if country_code in ['DZA', 'BWA', 'GAB', 'LBY', 'NAM', 'ZAF', 'TUN', 'EGY']:
        sigma = 0.06
    else: sigma = 0.25

    dist_probpep = ot.BetaMuSigma(df['probability_pep'].values[0],
                          sigma * (df['probability_pep_ub'].values[0] -df['probability_pep_lb'].values[0]) / 3.92,
                          df['probability_pep_lb'].values[0],
                          df['probability_pep_ub'].values[0]).getDistribution()


    # Define a copula for analysis
    copula = ot.IndependentCopula(2)

    # Define a composed distribution
    distribution = ot.ComposedDistribution([fittedDist, dist_probpep], copula)

    # Create a samploe from the composed distribution
    sample_res = distribution.getSample(1000000)

    # Define the relation for computing the number of lives lost
    f = ot.SymbolicFunction(['v0', 'X0'],
                            ['(1-X0)*v0'])

    # Calculate the result using the generated sample
    f_sample = f(sample_res)

    # Calculate the confidence intervals
    inf = f_sample.computeQuantile(.025)[0]
    mean = f_sample.computeMean()[0]
    sup = f_sample.computeQuantile(.975)[0]
    print('Mean value for the number of lives lost compared to the baseline: {:,.0f}'.format(mean))
    print(
        'Percentile interval [2.5%, 97.5%] for the number of lives lost compared to the baseline: '
        '[{:,.0f}, {:,.0f}]'.format(inf, sup))

    return [country_code, inf, mean, sup]

def compiling_data_lives_lost(folder_name:str='results'):
    # Calculation and compiling of confidence intervals for lives lost

    print("==================== Compiling data for the number of lives lost ======================")
    res = []

    logging.info("Starting compiling data for the number of lives lost")

    logging.info("Starting loop over countries")
    for country in countries_codes_list:
        res.append(estimate_confidence_interval_lives_lost(country, folder_name=folder_name))

    logging.info("Finished loop over countries")

    res_df = pd.DataFrame(data=res,
                 columns=['country_code', 'inf', 'mean', 'sup'])
    res_df.to_csv(
        '../results/res_lives_lost.csv',
        encoding= 'UTF-8',
        sep =';',
        decimal = '.')

    print('Mean value for the number of lives lost compared to the baseline: {:,.0f}'.format(res_df['mean'].sum()))
    print(
        'Percentile interval [2.5%, 97.5%] for the number of lives lost compared to the baseline: '
        '[{:,.0f}, {:,.0f}]'.format(res_df['inf'].sum(), res_df['sup'].sum()))

    logging.info("Finished")


def launch_post_treatment_process(process_number:int,
                                  folder_for_data_for_compiling:str="../results"):
    """ Launch the post treatment process.
    :param post_treatment_process_name: The name of the post treatment process to launch.

    1. Compile and save the results of simulations from the sensitivity analysis,
    as well as the individual plots for payoffs distributions

    2. Compile and save the results of simulations from the sensitivity
    analysis without individual plots

    3. Compile Sobol indices results and plot
    them (figures 1 and 2 in the appendix)

    4. Compile and save the data of the number of lives
    lost, as well as the total mean value and confidence interval

    5. Compile, save and visualize data from Monte-Carlo simulations for
    the dog population

    6. Visualize the payoffs distributions for all countries for the all vaccination (figure 3 in the appendix)
    and one vaccination (figure 4 in the appendix) strategies

    7. Realize the coalition analysis from the compiled
    data

    8. Visualize the yearly exposure to rabid dogs for different strategy profiles
    (figure 1 in the paper)

    9. Calculate and plot the break-even point for the vaccination strategy with and without accounting for the
    human capital effect (figure 7 in the appendix)

    10. Visualize the gains by country on the map
    (figures 2 and 3 in the paper)

    11. Calculate and plot the break-even point for different strategies with and without accounting for the
    human capital effect (figures 5 and 6 in the appendix)

    12. Plot and save total payoff distribution cumulative for all countries

    :return:
    """


    if process_number == 1:
        data_compile_save_and_plot(folder=f'{folder_for_data_for_compiling}/s_vaccination',
                                   save_summary=True,
                                   save_compiled_data=True,
                                   plot_individual_results=True)

        data_compile_save_and_plot(folder=f'{folder_for_data_for_compiling}/s_reintroduction',
                                   save_summary=True,
                                   save_compiled_data=True,
                                   plot_individual_results=True)
    elif process_number == 2:
        data_compile_save_and_plot(folder=f'{folder_for_data_for_compiling}/s_vaccination',
                                   save_summary=True,
                                   save_compiled_data=True,
                                   plot_individual_results=False)

        data_compile_save_and_plot(folder=f'{folder_for_data_for_compiling}/s_reintroduction',
                                   save_summary=True,
                                   save_compiled_data=True,
                                   plot_individual_results=False)

    elif process_number == 3:
        sobol_indices_compile_and_plot(folder=f'{folder_for_data_for_compiling}/s_vaccination')
        sobol_indices_compile_and_plot(folder=f'{folder_for_data_for_compiling}/s_reintroduction')

    elif process_number == 4:
        compiling_data_lives_lost(folder_name=folder_for_data_for_compiling)

    elif process_number == 5:
        compile_and_save_dog_population_data(folder_name=folder_for_data_for_compiling)

    elif process_number == 6:
        plot_comparing_payoffs('all_vac', folder_name=folder_for_data_for_compiling)
        plot_comparing_payoffs('one_vac', folder_name=folder_for_data_for_compiling)

    elif process_number == 7:
        coalition_analysis(folder_name=folder_for_data_for_compiling)

    elif process_number == 8:
        exposure_plot(folder_name=folder_for_data_for_compiling)

    elif process_number == 9:
        break_even_calculation(hce = False, folder_name=folder_for_data_for_compiling)
        break_even_calculation(hce = True, folder_name=folder_for_data_for_compiling)

    elif process_number == 10:
        visualize_gains(folder_name=folder_for_data_for_compiling)

    elif process_number == 11:
        cost_analysis_plot(hce = False, folder_name=folder_for_data_for_compiling)
        cost_analysis_plot(hce = True, folder_name=folder_for_data_for_compiling)

    elif process_number == 12:
        plot_and_save_total_payoff(folder_name=folder_for_data_for_compiling)

    else:
        raise ValueError("The process number must be between 1 and 12")


if __name__=='__main__':
    logging.basicConfig(level = logging.INFO)

    print("==================================== Running the post treatment process ====================================")
    print("The following processes are available:")
    print("1. Compile and save the results of simulations from the sensitivity analysis, as well as the individual plots for payoffs distributions")
    print("2. Compile and save the results of simulations from the sensitivity analysis without individual plots")
    print("3. Compile Sobol indices results and plot them (figures 1 and 2 in the appendix)")
    print("4. Compile and save the data of the number of lives lost, as well as the total mean value and confidence interval")
    print("5. Compile, save and visualize data from Monte-Carlo simulations for the dog population")
    print("6. Visualize the payoffs distributions for all countries for the all vaccination (figure 3 in the appendix) and one vaccination (figure 4 in the appendix) strategies")
    print("7. Realize the coalition analysis from the compiled data")
    print("8. Visualize the yearly exposure to rabid dogs for different strategy profiles (figure 1 in the paper)")
    print("9. Calculate and plot the break-even point for the vaccination strategy with and without accounting for the human capital effect (figure 7 in the appendix)")
    print("10. Visualize the gains by country on the map (figures 2 and 3 in the paper)")
    print("11. Calculate and plot the break-even point for different strategies with and without accounting for the human capital effect (figures 5 and 6 in the appendix)")
    print("12. Plot and save total payoff distribution cumulative for all countries")

    input_process_number = input("Please enter the number of the process you want to run: \n")

    try:
        process_number = int(input_process_number)
    except:
        raise ValueError("The input must be an integer")


    input_folder_name = input("If the data you want to compile or to use is in folder 'results', please enter '1'\n"
                              "If the data is in the folder 'original_results', please enter '2':\n")

    launch_post_treatment_process(process_number, '{}'.format('../results' if input_folder_name == '1' else '../original_results'))

    print("==================================== End of the post treatment process ====================================")