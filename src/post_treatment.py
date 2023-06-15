# Imports
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

# Imports from utils.py
from utils import  countries_names_dict, parameters_columns_names_list,parameters_columns_full_names_list, \
    countries_codes_list, summary_columns, coalition_pep, log_transform, experiment_strategy_from_folder

# Options
pd.set_option("display.expand_frame_repr", False)
warnings.simplefilter(action='ignore', category=FutureWarning)
fmt = '${x:,.0f}'


# Creation of results folders
outdirs = ['../results', '../results/s_vaccination', '../results/s_reintroduction', '../img', '../img/inputs_dist',
           '../img', '../results/all_vac', '../results/one_vac']
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

    # Get the strategy name from the folder name
    experiment_strategy = experiment_strategy_from_folder(folder)

    # Create DataFrames for saving the results
    all_first_sobol_indices = pd.DataFrame(columns=parameters_columns_names_list)
    all_total_sobol_indices = pd.DataFrame(columns=parameters_columns_names_list)

    # Loop over the countries to compile the data
    for country_code in countries_codes_list:

        df = pd.read_csv(f'{folder}/sensitivity_{country_code}_{experiment_strategy}.csv',
                         encoding='UTF-8',
                         sep = ';',
                         decimal='.')

        df.rename(columns = {'output_all_first_order' : 'first_order',
                             'output_all_total_order' : 'total_order',
                             'Unnamed: 0' : 'variable'},
                  inplace = True)

        all_first_sobol_indices = all_first_sobol_indices.append(pd.DataFrame(data = [np.concatenate(([country_code],df['first_order'].values))],
                                                                              columns= parameters_columns_names_list),
                                                                 ignore_index=True)
        all_total_sobol_indices = all_total_sobol_indices.append(pd.DataFrame(data = [np.concatenate(([country_code],df['total_order'].values))],
                                                                              columns= parameters_columns_names_list),
                                                                 ignore_index=True)


    # Transforming values into float
    all_first_sobol_indices[parameters_columns_names_list[1:]] = all_first_sobol_indices[parameters_columns_names_list[1:]].astype(float)
    all_total_sobol_indices[parameters_columns_names_list[1:]] = all_total_sobol_indices[parameters_columns_names_list[1:]].astype(float)

    # Capping negative values at 0. Sobol indices are positive, the confidence interval is around zero and some values can appear negative
    all_first_sobol_indices[parameters_columns_names_list[1:]] = np.where(all_first_sobol_indices[parameters_columns_names_list[1:]] < 0, 0, all_first_sobol_indices[parameters_columns_names_list[1:]])
    all_total_sobol_indices[parameters_columns_names_list[1:]] = np.where(all_total_sobol_indices[parameters_columns_names_list[1:]] < 0, 0, all_total_sobol_indices[parameters_columns_names_list[1:]])

    # Saving the resulting DataFrames
    if save :
        all_first_sobol_indices.to_csv(f'first_order_sobol_{experiment_strategy}.csv', encoding ='UTF-8', sep = ';', decimal='.')
        all_total_sobol_indices.to_csv(f'total_order_sobol_{experiment_strategy}.csv', encoding ='UTF-8', sep = ';', decimal='.')

    # Ploting the Sobol indices
    if plot :
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
        ax1.set_title("First order Sobol' indices")

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
        ax2.set_title("Total order Sobol' indices")
        plt.show()

        fig.savefig(f'../img/{experiment_strategy}/sobol_indices_{experiment_strategy}.png',
                    dpi='figure',
                    format=None,
                    metadata=None,
                    bbox_inches=None,
                    pad_inches=0.1,
                    facecolor='auto',
                    edgecolor='auto',
                    backend=None)

def data_compile_save_and_plot(folder:str = '../results/s_vaccination',
                                save_summary:bool = True,
                                save_compiled_data:bool = True,
                                plot_individual_results:bool = True,
                                plot_payoffs_comparing:bool = True):
    """
    Compile data from payoffs calculations and summarize. Then, plot the data at individual level or all countries.

    :param folder: Folder containing the data.
    :param save_summary: Save the summary of results.
    :param save_compiled_data: Save the compiled data.
    :param plot_individual_results: Plot the individual results.
    :param plot_payoffs_comparing: Plot all results.
    :return:
    """

    # Initializing resulting dataframes
    data = pd.DataFrame(columns=['country_code', 'benefice_from_baseline', 'cumulated_payoff', 'sim_id'])
    summary_data = pd.DataFrame(columns=summary_columns)

    # Get the strategy profile name from folder name
    experiment_strategy = experiment_strategy_from_folder(folder)

    # Loop over the countries for compiling the results
    for country_code in countries_codes_list:
        df = pd.read_csv(f'{folder}/simulation_results_{country_code}_{experiment_strategy}.csv',
                         encoding='UTF-8',
                         sep = ';',
                         decimal='.')

        df['country_code'] = country_code
        df['baseline_payoff'] = df['cumulated_payoff'] - df['benefice_from_baseline']

        nbr_negative_ben = np.sum(df['benefice_from_baseline']<0)
        nombre_sim = len(df['benefice_from_baseline'])

        print('{} : Quantiles [2.5%, 97.5%] for gains/losses compared to baseline: [{}, {}]'.format(country_code, df['benefice_from_baseline'].quantile(0.025),
                                                                                                df['benefice_from_baseline'].quantile(0.975)))
        # Summary
        summary_data = summary_data.append(pd.DataFrame(data = [[
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
            columns = summary_columns),
            ignore_index=False
        )

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
            fig.savefig(f'../img/{experiment_strategy}/payoff_distribution_{country_code}_{experiment_strategy}.png',
                        dpi='figure',
                        format=None,
                        metadata=None,
                        bbox_inches=None,
                        pad_inches=0.1,
                        facecolor='auto',
                        edgecolor='auto',
                        backend=None)

    # Save summary data
    if save_summary :
        summary_data.to_csv(f'../results/summary_data_{experiment_strategy}.csv', encoding ='UTF-8', sep = ';', decimal='.')

    # Save compiled data
    if save_compiled_data:
        data.to_csv(f'../results/compiled_data_{experiment_strategy}.csv', encoding ='UTF-8', sep = ';', decimal='.')

    # Plot compared data
    if plot_payoffs_comparing:
        del data

        summary_data.sort_values(by = 'gain_loss_to_baseline_mean', inplace = True)

        fig, axes = plt.subplots(2,1)

        x_pos = np.arange(0,48,1)

        # y_err = np.stack([summary_data['gain_loss_to_baseline_0025'].values,
        #                   summary_data['gain_loss_to_baseline_0975'].values],
        #                  axis = 1)
        #
        # y_err_log = np.stack([summary_data['gain_loss_to_baseline_0025_log'].values,
        #                   summary_data['gain_loss_to_baseline_0975_log'].values],
        #                  axis = 1)

        fig.suptitle(
            f"Countries {experiment_strategy} strategy relative gains/losses with 2.5% and 97.5% percentiles (linear and log scales)")

        sns.barplot(data = summary_data,
                    x = 'country_code',
                    y ='gain_loss_to_baseline_mean',
                    ax = axes[0],
                    palette = sns.color_palette("viridis", n_colors = 48, as_cmap=False))
        
        axes[0].vlines(x_pos,
                       summary_data['gain_loss_to_baseline_0025'].values,
                       summary_data['gain_loss_to_baseline_0975'].values,
                       colors = 'r',
                       linestyles = 'solid')

        tick = mtick.StrMethodFormatter(fmt)
        axes[0].yaxis.set_major_formatter(tick)
        axes[0].set_xticks([])
        axes[0].set(ylabel='Relative gains/losses (in $)', xlabel = None)

        sns.barplot(data = summary_data,
                    x = 'country_code',
                    y ='gain_loss_to_baseline_mean_log',
                    ax = axes[1],
                    palette =  sns.color_palette("viridis", n_colors = 48, as_cmap=False)
                    )

        axes[1].vlines(x_pos,
                       summary_data['gain_loss_to_baseline_0025_log'].values,
                       summary_data['gain_loss_to_baseline_0975_log'].values,
                       colors = 'r',
                       linestyles = 'solid')
        axes[1].set(ylabel= 'Special log transformation of gains/losses (cf. methodology)', xlabel = 'Country ISO-3 code' )
        plt.show()

        fig.savefig(f'../img/{experiment_strategy}/countries_payoffs_{experiment_strategy}.png',
                    dpi='figure',
                    format=None,
                    metadata=None,
                    bbox_inches=None,
                    pad_inches=0.1,
                    facecolor='auto',
                    edgecolor='auto',
                    backend=None)

def plot_and_save_total_payoff():
    """
    Plot the total payoffs distributions
    :return:
    """

    # Read the data
    df = pd.read_csv('../results/compiled_data_all_vac.csv', encoding ='UTF-8', sep = ';', decimal='.')

    # print(df.head(10))
    groupby_sim = df.groupby('sim_id')[['benefice_from_baseline', 'cumulated_payoff']].sum()
    groupby_sim['baseline_payoff'] = groupby_sim['cumulated_payoff'] - groupby_sim['benefice_from_baseline']
    # print(groupby_sim)
    print('Mean value for the gains/losses compared to the baseline : ${:,.0f}'.format(groupby_sim['benefice_from_baseline'].mean()))
    print('Percentile interval [2.5%, 97.5%] for the gains/losses compared to the baseline: [${:,.0f}, ${:,.0f}]'.format(groupby_sim['benefice_from_baseline'].quantile(q = 0.025),
                                                                                                           groupby_sim['benefice_from_baseline'].quantile(q = 0.975)))
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
    fig.savefig(f'../img/payoff_distribution_all_vac_all_countries.png',
                dpi='figure',
                format=None,
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto',
                backend=None)

def compile_and_save_dog_population_data():
    """
    Compile the resulting data from the Monte Carlo simulations for rabid dog population, exposed humans and clinical cases.

    :return:
    """


    df = pd.DataFrame(columns = ['country_code', 'rabid_dogs_population', 'exposed_humans', 'clinical_cases', 'total_dog_population','sim_id'])

    # Loop over the countries for compiling
    for country in countries_codes_list:
        inter = pd.read_csv(f'../results/dog_pop/simulation_dog_population_results_{country}.csv', encoding = 'UTF-8', sep = ';', decimal = '.', index_col=0)
        inter['sim_id'] = range(200000)
        df = pd.concat([df, inter], axis = 0,ignore_index = True)

    # Save the compiled data
    df.to_csv('../results/compiled_data_rabid_dog_population.csv', encoding='UTF-8', sep =';', decimal = '.')

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
    fig.savefig(f'../img/rabid_dog_population_distribution_baseline.png',
                dpi='figure',
                format=None,
                metadata=None,
                bbox_inches=None,
                pad_inches=0.1,
                facecolor='auto',
                edgecolor='auto',
                backend=None)

def coalition_analysis():
    """
    Coalition analysis with confidence interval. Payoffs comparison between different strategies possibilities.
    :return:
    """

    def coalition_group(row) :
        if row in coalition_pep:
            return 'C_PEP'
        else:
            return 'C_VAC'

    # Read data
    df = pd.read_csv('../results/compiled_data_all_vac.csv', encoding ='UTF-8', sep = ';', decimal='.')

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
    df = pd.read_csv('../results/compiled_data_one_vac.csv', encoding ='UTF-8', sep = ';', decimal='.')

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

def visualize_gains():
    """
    Visualize gains on the map of Africa.
    :return:
    """

    # Load shapefile of African countries
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    africa = world.query('continent == "Africa"')

    # Load GDP data
    gdp_data = pd.read_excel("../data/summary_data_for_simulation.xlsx",
                                     sheet_name = "Pop_Data",
                                     header = 0)

    gdp_data['nash_strategy'] = np.array([
            0,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1
        ])

    gains_data = pd.read_csv('../results/summary_data_all_vac.csv', encoding='UTF-8', sep = ';', decimal= ',')


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
    africa_map['gains_by_gdp'] = africa_map['gain_loss_to_baseline_mean']/africa_map['gdp_2024']
    print(africa_map)
    # Set up the plot

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

def exposure_plot():
    """
    Plot the number of exposed humans over the years under different scenarios.
    :return:
    """

    files_names = ['ALL_PEP', 'ALL_VAC', 'NASH', 'number_24', 'one_pep']
    res = pd.DataFrame(data = {"year" :  range(2024, 2055)})

    # Read data
    for file_name in files_names:
        df = pd.read_csv(f"../results/results_by_year_exposed/exposed_humans_{file_name}.csv",
                         encoding = 'UTF-8',
                         sep = ';',
                         decimal = ',' if file_name in ['number_24', 'one_pep'] else '.',
                         index_col = 0)

        df[file_name.lower()] = np.sum(df[countries_codes_list], axis=1)

        if file_name in ['number_24', 'one_pep']:

            df = df.groupby('year', as_index=False)[file_name].mean()
            print(df)

        res = pd.concat([res, df[[file_name.lower()]]], axis = 1)

    initial_pop =  398529 / (1+0.012694042)

    res = pd.concat([
        pd.DataFrame(data =  [[2023, initial_pop, initial_pop, initial_pop, initial_pop, initial_pop]],
                     columns = ['year', 'all_pep', 'all_vac', 'nash', 'number_24', 'one_pep']),
        res
    ], axis=0, ignore_index=True)

    print(res)
    fig, ax = plt.subplots(1,1)
    plt.title("Total Number of Humans Exposed to Rabid Dogs per Year for all 48 countries depending on the strategy profile")
    plt.plot(res['year'], res['all_vac'], color = 'g', linestyle = '-', marker = 'x', label = 'Vaccination campaign in all countries')
    plt.plot(res['year'], res['all_pep'], color = 'r', linestyle = '-', marker = 'x', label = 'Baseline (No vaccination)')
    plt.plot(res['year'], res['nash'], color = 'k', linestyle = '--', marker = 'x', label = 'Nash equilibrium (vaccination where it is a dominant strategy)')
    plt.plot(res['year'], res['one_pep'], color = 'c', linestyle = '--', marker = '.', label = 'Average for strategy profile with one defecting country')
    plt.plot(res['year'], res['number_24'], color = 'b', linestyle = '--', marker = '.', label = 'Average for strategy profile with half vaccinated')
    plt.legend()
    plt.grid()
    plt.ylabel('Number of Humans Exposed')
    plt.xlabel('Year')
    fmt = '{x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    plt.show()

def break_even_calculation(hce = False):
    """
    Plot breakeven years with and without consideration for HCE.
    :param hce: Include HCE or not.
    :return:
    """

    # res = pd.DataFrame(data={"year": range(2024, 2055)})

    if hce:
        file_name_prefix = 'payoff_total'
    else :
        file_name_prefix = 'payoff_without_hce'

    # Read data
    df_pep = pd.read_csv(f"../results/results_by_year/{file_name_prefix}_ALL_PEP.csv",
                         encoding='UTF-8',
                         sep=';',
                         decimal='.',
                         index_col=0)

    df_vac = pd.read_csv(f"../results/results_by_year/{file_name_prefix}_ALL_VAC.csv",
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
    fig, ax = plt.subplots()

    ax.set_title("Breakeven point per country")
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

def cost_analysis_plot(hce = True):
    """
    Plot the cost analysis over the years.
    :param hce:
    :return:
    """

    files_names = ['ALL_PEP', 'ALL_VAC', 'NASH', 'n24', 'one_pep']
    res = pd.DataFrame(data={"year": range(2024, 2055)})

    # Read the data
    for file_name in files_names:
        if hce:
            file_name_prefix = 'payoff_total'
        else :
            file_name_prefix = 'payoff_without_hce'

        df = pd.read_csv(f"../results/results_by_year/{file_name_prefix}_{file_name}.csv",
                         encoding='UTF-8',
                         sep=';',
                         decimal=',' if file_name in ['n24', 'one_pep'] else '.',
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

    # Calculate the cumulative payoffs
    res_cumsum = res.copy()
    res_cumsum[['all_pep', 'all_vac', 'nash', 'n24', 'one_pep']] = res[['all_pep', 'all_vac', 'nash', 'n24', 'one_pep']].cumsum()

    # Plot
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


def estimate_confidence_interval_lives_lost(country_code):
    """
    Estimate the confidence interval for the number of lives lost due to rabies at the baseline.
    :param country_code: The country code of the country we want to estimate the confidence interval.
    :return:
    """

    print("==================== Running Monte Carlo for the number of lives lost for {} ======================".format(country_code))

    # Read the data for the probability of receiving PEP
    df = pd.read_excel("../data/summary_data_for_simulation.xlsx", #./Alvar/Dog Population Knobel 2005 including PopGrowth.xlsx
                       sheet_name="Pop_Data",
                       header=0)
    df = df[df['country_code']==country_code]

    # Read the data from Monte Carlo simulations
    df_res = pd.read_csv(f'../results/dog_pop/simulation_dog_population_results_{country_code}.csv', encoding='UTF-8', sep = ';', decimal='.')

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
    inf = f_sample.computeQuantile(.025)
    mean = f_sample.computeMean()
    sup = f_sample.computeQuantile(.975)

    print('Mean value for the gains/losses (ONE VAC) compared to the baseline for coalition C_PEP: ${:,.0f}'.format(mean))
    print(
        'Percentile interval [2.5%, 97.5%] for the gains/losses compared to the baseline coalition C_PEP: '
        '[${:,.0f}, ${:,.0f}]'.format(inf, sup))

    return [country_code, inf, mean, sup]

    

if __name__=='__main__':
    # compile_and_save_dog_population_data()

    # coalition_analysis()

    visualize_gains()

    # exposure_plot()
    #
    # break_even_calculation(False)
    # break_even_calculation(True)
    #
    # cost_analysis_plot(hce=False)
    # cost_analysis_plot(hce=True)
    #
    # sobol_indices_compile_and_plot(folder = '../results/s_vaccination',save = False, plot = True)
    # sobol_indices_compile_and_plot(folder = '../results/s_reintroduction',save = False, plot = True)
    #
    # data_compile_save_and_plot(folder='../results/s_vaccination',
    #                            save_summary=False,
    #                            save_compiled_data=False,
    #                            plot_individual_results=False,
    #                            plot_payoffs_comparing=False)
    #
    # data_compile_save_and_plot(folder='../results/s_reintroduction',
    #                            save_summary=False,
    #                            save_compiled_data=False,
    #                            plot_individual_results=False,
    #                            plot_payoffs_comparing=False)
    #
    #
    # # Calculation and compiling of confidence intervals for lives lost
    # res = []
    #
    # for country in countries_codes_list:
    #     res.append(estimate_confidence_interval_lives_lost(country))
    #
    # pd.DataFrame(data=res,
    #              columns=['country_code', 'inf', 'mean', 'sup']).to_csv(
    #           'res_lives_lost.csv',
    #           encoding= 'UTF-8',
    #           sep =';',
    #           decimal = '.')
