# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:21:08 2021

@author: allan
"""

import grape
import algorithms
from functions import add, sub, mul, pdiv, neg, and_, or_, not_, less_than_or_equal, greater_than_or_equal

from os import path
import pandas as pd
import numpy as np
from deap import creator, base, tools
import random

from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt
import os

problem = 'heartDisease'

def setDataSet(problem, RANDOM_SEED):
    np.random.seed(RANDOM_SEED)
    if problem == 'australian': #66
        data =  pd.read_csv(r"datasets/australian.dat", sep=" ")    
        l = data.shape[0]
        Y = np.zeros([l,], dtype=int)
        for i in range(l):
            Y[i] = data['output'].iloc[i]
        data.pop('output')
        #continuous features: d1, d2, d6, d9, d12, d13
        #categorical features:
        #d0: two
        #d3: three => change 3 to 0
        #data['d3'] = data['d3'].replace([3], 0)
        #d4: 14 => change 14 to 0
        #data['d4'] = data['d4'].replace([14], 0)
        #d5: 9 => change 9 to 0
        #data['d5'] = data['d5'].replace([9], 0)
        #d7: two
        #d8: two
        #d10: two
        #d11: three => change 3 to 0
        #data['d11'] = data['d11'].replace([3], 0)     
        
        #Convert categorical using one-hot enconding
        dataOneHot = pd.get_dummies(data, columns=['d3', 'd4', 'd5', 'd11'])
        
        X = dataOneHot.to_numpy()
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=RANDOM_SEED)
        
        X_train = np.transpose(X_train)
        X_test = np.transpose(X_test)
        
        GRAMMAR_FILE = 'australian.bnf'
    
    if problem == 'carEvaluation':
        Y = np.zeros([1727,], dtype=int)
    
        column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
        
        data = pd.read_csv(r"datasets/car.data", sep=",", header=0, names=column_names)
        
        for i in range(1727):
            if data['class'].iloc[i] == 'unacc':
                Y[i] = 0
            elif data['class'].iloc[i] == 'acc':
                Y[i] = 1
            elif data['class'].iloc[i] == 'good':
                Y[i] = 2
            elif data['class'].iloc[i] == 'vgood':
                Y[i] = 3
            
        data = data.drop(['class'], axis=1)
        
        #Using oneHot encoding on categorical (non binary) features
        dataOneHot = pd.get_dummies(data)
        
        X = dataOneHot.to_numpy()
            
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=RANDOM_SEED)
        
        X_train = np.transpose(X_train)
        X_test = np.transpose(X_test)
        
        GRAMMAR_FILE = 'carEvaluation.bnf'
    
    if problem == 'Banknote':
        #There 1813 samples with class 1
        #We'll split into 70% for training and 30% for test, assuring the balanced data
        X_train = np.zeros([1000, 4], dtype=float)
        Y_train = np.zeros([1000,], dtype=bool)
        X_test = np.zeros([372, 4], dtype=float)
        Y_test = np.zeros([372,], dtype=bool)
    
        data = pd.read_table(r"datasets/banknote_Train.csv", sep=" ")
        for i in range(1000):
            for j in range(4):
                X_train[i,j] = data['x'+ str(j)].iloc[i]
        for i in range(1000):
            Y_train[i] = data['y'].iloc[i] > 0
            
        data = pd.read_table(r"datasets/banknote_Test.csv", sep=" ")
        for i in range(372):
            for j in range(4):
                X_test[i,j] = data['x'+ str(j)].iloc[i]
        for i in range(372):
            Y_test[i] = data['y'].iloc[i] > 0
        
        X_train = np.transpose(X_train)
        X_test = np.transpose(X_test)
            
        GRAMMAR_FILE = 'Banknote.bnf'
        
    if problem == 'spambase':
        #There 1813 samples with class 1
        #We'll split into 70% for training and 30% for test, assuring the balanced data
        X = np.zeros([4601, 57], dtype=float)
        Y = np.zeros([4601,], dtype=int)
    
        data = pd.read_table(r"datasets/spambase.csv")
        for i in range(4601):
            for j in range(57):
                X[i,j] = data['d'+ str(j)].iloc[i]
        for i in range(4601):
            Y[i] = data['class'].iloc[i]
            
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=RANDOM_SEED)
        
        X_train = np.transpose(X_train)
        X_test = np.transpose(X_test)
            
        GRAMMAR_FILE = 'spambase.bnf'
    
    if problem == 'heartDisease':
        data =  pd.read_csv(r"datasets/processed.cleveland.data", sep=",")
        #There are some data missing on columns d11 and d12, so let's remove the rows
        data = data[data.ca != '?']
        data = data[data.thal != '?']
        
        #There are 160 samples with class 0, 54 with class 1, 35 with class 2,
        #35 with class 3 and 13 with class 4
        #Let's consider the class 0 and all the remaining as class 1
        Y = data['class'].to_numpy()
        for i in range(len(Y)):
            Y[i] = 1 if Y[i] > 0 else 0
        data = data.drop(['class'], axis=1)
        
        data.loc[:, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = (data.loc[:, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] - data.loc[:, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].mean())/data.loc[:, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].std()
        
        data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])#, prefix = ['cp']) = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])#, prefix = ['cp'])
        
        X = data.to_numpy()
      
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=RANDOM_SEED)
        
        X_train = np.transpose(X_train)
        X_test = np.transpose(X_test)
            
        GRAMMAR_FILE = 'heartDisease.bnf'
    
    BNF_GRAMMAR = grape.Grammar(r"grammars/" + GRAMMAR_FILE)

    return X_train, Y_train, X_test, Y_test, BNF_GRAMMAR

def mae(y, yhat):
    """
    Calculate mean absolute error between inputs.

    :param y: The expected input (i.e. from dataset).
    :param yhat: The given input (i.e. from phenotype).
    :return: The mean absolute error.
    """
    
    compare = np.equal(y,yhat)

    return 1 - np.mean(compare)

def fitness_eval(individual, points):
    x = points[0]
    Y = points[1]
    
    if individual.invalid == True:
        return np.nan,

    # Evaluate the expression
    try:
        pred = eval(individual.phenotype)
    except (FloatingPointError, ZeroDivisionError, OverflowError,
            MemoryError):
        # FP err can happen through eg overflow (lots of pow/exp calls)
        # ZeroDiv can happen when using unprotected operators
        return np.nan,
    assert np.isrealobj(pred)
    
    try:
        Y_class = [1 if pred[i] > 0 else 0 for i in range(len(Y))]
    except (IndexError, TypeError):
        return np.nan,
    fitness = mae(Y, Y_class)
    
    return fitness,

def plot_individual_runs(all_max_fitness, all_mean_fitness, all_min_fitness, all_fitness_test, n_runs):
    """Plot individual runs in a subplot grid"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for run in range(n_runs):
        if run < len(axes):
            ax = axes[run]
            generations = range(len(all_max_fitness[run]))
            
            # Plot training fitness
            ax.plot(generations, all_max_fitness[run], 'r-', label='Best Training', linewidth=2)
            ax.plot(generations, all_mean_fitness[run], 'b-', label='Avg Training', linewidth=2)
            ax.plot(generations, all_min_fitness[run], 'g-', label='Min Training', linewidth=2)
            
            # Plot test fitness
            ax.plot(generations, all_fitness_test[run], 'm--', label='Test Fitness', linewidth=2)
            
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.set_title(f'Run {run + 1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grape/results/example_classification/individual_runs_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_single_run_detailed(max_fitness, mean_fitness, min_fitness, fitness_test, run_number=1):
    """Plot detailed view of a single run with error bars"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    generations = range(len(max_fitness))
    
    # Plot training fitness with different styles
    ax.plot(generations, max_fitness, 'r-', label='Best Training', linewidth=3)
    ax.plot(generations, mean_fitness, 'b-', label='Average Training', linewidth=2)
    ax.plot(generations, min_fitness, 'g-', label='Min Training', linewidth=2)
    
    # Plot test fitness
    ax.plot(generations, fitness_test, 'm--', label='Test Fitness', linewidth=3)
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_title(f'Detailed Fitness Evolution - Run {run_number}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add statistics box
    stats_text = f'Final Best Training: {max_fitness[-1]:.4f}\n'
    stats_text += f'Final Average Training: {mean_fitness[-1]:.4f}\n'
    stats_text += f'Final Test Fitness: {fitness_test[-1]:.4f}\n'
    stats_text += f'Improvement: {max_fitness[0] - max_fitness[-1]:.4f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'grape/results/example_classification/run_{run_number}_detailed_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)

toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual) 
#toolbox.register("populationCreator", grape.random_initialisation, creator.Individual) 
#toolbox.register("populationCreator", grape.PI_Grow, creator.Individual) 

toolbox.register("evaluate", fitness_eval)

# Tournament selection:
toolbox.register("select", tools.selTournament, tournsize=7) #selLexicaseFilter

# Single-point crossover:
toolbox.register("mate", grape.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", grape.mutation_int_flip_per_codon)

POPULATION_SIZE = 500
MAX_INIT_TREE_DEPTH = 13
MIN_INIT_TREE_DEPTH = 4

MAX_GENERATIONS = 200
P_CROSSOVER = 0.8
P_MUTATION = 0.01
ELITE_SIZE = 0#round(0.01*POPULATION_SIZE) #it should be smaller or equal to HALLOFFAME_SIZE
HALLOFFAME_SIZE = 1#round(0.01*POPULATION_SIZE) #it should be at least 1

MIN_INIT_GENOME_LENGTH = 95#*6
MAX_INIT_GENOME_LENGTH = 115#*6
random_initilisation = False #put True if you use random initialisation

CODON_CONSUMPTION = 'lazy'
GENOME_REPRESENTATION = 'list'
MAX_GENOME_LENGTH = None#'auto'

MAX_TREE_DEPTH = 35 #equivalent to 17 in GP with this grammar
MAX_WRAPS = 0
CODON_SIZE = 255

REPORT_ITEMS = ['gen', 'invalid', 'avg', 'std', 'min', 'max', 
                'fitness_test',
                'best_ind_length', 'avg_length', 
                'best_ind_nodes', 'avg_nodes', 
                'best_ind_depth', 'avg_depth', 
                'avg_used_codons', 'best_ind_used_codons', 
                'structural_diversity', 'fitness_diversity',
                'selection_time', 'generation_time']

N_RUNS = 6

# Lists to store data from all runs for plotting
all_max_fitness = []
all_mean_fitness = []
all_min_fitness = []
all_fitness_test = []

for i in range(N_RUNS):
    print()
    print()
    print("Run:", i)
    print()
    
    RANDOM_SEED = i + 1
    
    X_train, Y_train, X_test, Y_test, BNF_GRAMMAR = setDataSet(problem, RANDOM_SEED) #We set up this inside the loop for the case in which the data is defined randomly

    random.seed(RANDOM_SEED) 

    # create initial population (generation 0):
    if random_initilisation:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE, 
                                           bnf_grammar=BNF_GRAMMAR, 
                                           min_init_genome_length=MIN_INIT_GENOME_LENGTH,
                                           max_init_genome_length=MAX_INIT_GENOME_LENGTH,
                                           max_init_depth=MAX_TREE_DEPTH, 
                                           codon_size=CODON_SIZE,
                                           codon_consumption=CODON_CONSUMPTION,
                                           genome_representation=GENOME_REPRESENTATION
                                           )
    else:
        population = toolbox.populationCreator(pop_size=POPULATION_SIZE, 
                                           bnf_grammar=BNF_GRAMMAR, 
                                           min_init_depth=MIN_INIT_TREE_DEPTH,
                                           max_init_depth=MAX_INIT_TREE_DEPTH,
                                           codon_size=CODON_SIZE,
                                           codon_consumption=CODON_CONSUMPTION,
                                           genome_representation=GENOME_REPRESENTATION
                                            )
    
    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALLOFFAME_SIZE)
    
    # prepare the statistics object:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)
    
    # perform the Grammatical Evolution flow:
    population, logbook = algorithms.ge_eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, elite_size=ELITE_SIZE,
                                              bnf_grammar=BNF_GRAMMAR, 
                                              codon_size=CODON_SIZE, 
                                              max_tree_depth=MAX_TREE_DEPTH,
                                              max_genome_length=MAX_GENOME_LENGTH,
                                              points_train=[X_train, Y_train], 
                                              points_test=[X_test, Y_test], 
                                              codon_consumption=CODON_CONSUMPTION,
                                              report_items=REPORT_ITEMS,
                                              genome_representation=GENOME_REPRESENTATION,                                              
                                              stats=stats, halloffame=hof, verbose=False)
    
    import textwrap
    best = hof.items[0].phenotype
    print("Best individual: \n","\n".join(textwrap.wrap(best,80)))
    print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
    
    print("Depth: ", hof.items[0].depth)
    print("Length of the genome: ", len(hof.items[0].genome))
    print(f'Used portion of the genome: {hof.items[0].used_codons/len(hof.items[0].genome):.2f}')
    
    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
    min_fitness_values, std_fitness_values = logbook.select("min", "std")
    fitness_test = logbook.select("fitness_test")
    
    # Store data for plotting
    all_max_fitness.append(max_fitness_values)
    all_mean_fitness.append(mean_fitness_values)
    all_min_fitness.append(min_fitness_values)
    all_fitness_test.append(fitness_test)
    
    best_ind_length = logbook.select("best_ind_length")
    avg_length = logbook.select("avg_length")

    selection_time = logbook.select("selection_time")
    generation_time = logbook.select("generation_time")
    gen, invalid = logbook.select("gen", "invalid")
    avg_used_codons = logbook.select("avg_used_codons")
    best_ind_used_codons = logbook.select("best_ind_used_codons")
    
    best_ind_nodes = logbook.select("best_ind_nodes")
    avg_nodes = logbook.select("avg_nodes")

    best_ind_depth = logbook.select("best_ind_depth")
    avg_depth = logbook.select("avg_depth")

    structural_diversity = logbook.select("structural_diversity") 
    fitness_diversity = logbook.select("fitness_diversity")     
    
    r = RANDOM_SEED    
    header = REPORT_ITEMS
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs("grape/results/example_classification", exist_ok=True)
    
    with open(r"grape/results/example_classification/" + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(header)
        for value in range(len(max_fitness_values)):
            writer.writerow([gen[value], invalid[value], mean_fitness_values[value],
                             std_fitness_values[value], min_fitness_values[value],
                             max_fitness_values[value], 
                             fitness_test[value],
                             best_ind_length[value], 
                             avg_length[value], 
                             best_ind_nodes[value],
                             avg_nodes[value],
                             best_ind_depth[value],
                             avg_depth[value],
                             avg_used_codons[value],
                             best_ind_used_codons[value], 
                             structural_diversity[value],
                             fitness_diversity[value],
                             selection_time[value], 
                             generation_time[value]])

# After all runs are completed, create plots
print("\n" + "="*50)
print("CREATING PLOTS")
print("="*50)

# Plot individual runs
plot_individual_runs(all_max_fitness, all_mean_fitness, all_min_fitness, all_fitness_test, N_RUNS)

# Plot detailed view of Run 1
plot_single_run_detailed(all_max_fitness[0], all_mean_fitness[0], all_min_fitness[0], all_fitness_test[0], 1)

# Print summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)

best_final_fitness = min([run[-1] for run in all_max_fitness])
avg_final_fitness = np.mean([run[-1] for run in all_max_fitness])
best_run_idx = np.argmin([run[-1] for run in all_max_fitness])

print(f"Best final training fitness across all runs: {best_final_fitness:.4f}")
print(f"Average final training fitness across all runs: {avg_final_fitness:.4f}")
print(f"Best run (lowest final fitness): Run {best_run_idx + 1}")
print(f"Number of runs completed: {N_RUNS}")

# Print test fitness summary
test_final_fitness = [run[-1] for run in all_fitness_test]
best_test_fitness = min(test_final_fitness)
avg_test_fitness = np.mean(test_final_fitness)
best_test_run_idx = np.argmin(test_final_fitness)

print(f"\nBest final test fitness across all runs: {best_test_fitness:.4f}")
print(f"Average final test fitness across all runs: {avg_test_fitness:.4f}")
print(f"Best test run: Run {best_test_run_idx + 1}")

print(f"\nAll plots and CSV files saved in: grape/results/example_classification/")