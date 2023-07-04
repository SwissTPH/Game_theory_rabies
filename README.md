# Game_theory_rabies

Code used for the simulations, data analysis and sensitivity analysis for the article "The potential coordination 
benefits in disease control - evidence from rabies elimination efforts in Africa"

## List of appendices and files

- Appendix 0 (Word): Notations and descriptions
- Appendix 1 (Word): Methods
- Appendix 2 (Word): Modelling of the Game "Coordinated dog rabies vaccination against incomplete PEP"
- Appendix 3 (Word): The different layers of the game
- Appendix 4 (Excel file): `Appendix 4 - Data for simulations.xlsx`
- Appendix 5 (Excel file): `Appendix 5 - Strategy analysis.xlsx`
- Appendix 6 (Excel file): `Appendix 6 - Sobol indices.xlsx`
- Appendix 7 (Excel file): `Appendix 7 - Neighbouring countries matrix.xlsx`
- Appendix 8 (CSV file): `Appendix 8 - Minimal distance matrix.csv`
- Appendix 9 (Excel file): `Appendix 9 - GDP Contribution Calculation 20220902.xlsx`
- Appendix 10 (Excel file): `Appendix 10_1 - Dog Population Knobel 2005 including PopGrowth.xlsx`,
`Appendix 10_2 - pntd.0003709.s002_scoping.xlsx`

+ Results from simulations: folder `results`
+ Generated figures: folder `img`
+ Python scripts:
  + `utils.py`
  + `monte_carlo_baseline_populations.py`
  + `strategy_analysis.py`
  + `sensitivity_analysis_model_one_country.py`
  + `post_treatment.py`

## Installation and usage
### Setting up the environment
#### 1. For Windows users
For Windows users, we recommend to install [Anaconda](https://www.anaconda.com/products/individual) to manage 
the installation of the required packages. 
1. Download and install Anaconda
2. Download the repository from GitHub ([Game_theory_rabies](https://github.com/SwissTPH/Game_theory_rabies))
3. Open the Anaconda prompt and navigate to the folder where the repository is located
4. Create a virtual environment named `game_theory_rabies` using the following command
```anaconda prompt
conda create -n game_theory_rabies python=3.8
```
5. Activate the environment
```anaconda prompt
conda activate game_theory_rabies
```
6. Install the required packages
```anaconda prompt
pip install -r requirements.txt
```
#### 2. For Linux users

For Linux users, we advise to create a virtual environment using `venv` with
the python version 3.8.
1. Download the repository from GitHub ([Game_theory_rabies](https://github.com/SwissTPH/Game_theory_rabies))
2. Open a terminal and navigate to the folder where the repository is located
3. Install Python 3.8, if not already installed (/!\ Admin rights are required)
```commandline
sudo apt-get install python3.8
```
4. Create a virtual environment named `game_theory_rabies` using the following command
```commandline
python3.8 -m venv game_theory_rabies
```
5. Activate the environment
```commandline
source game_theory_rabies/bin/activate
```
6. Install the required packages
```commandline
pip install -r requirements.txt
```

### Running the scripts
#### 1. Strategy analysis



#### 2. Sensitivity analysis





