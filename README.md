# Bayesian optimization for dynamic pathway engineering

Python code for optimizing genetic control architectures and parameter values simultaneously using Hyperopt. Data and figure files under the Github size limit are included; large data files are listed and available upon request. This code accompanies the recent paper:

*A machine learning approach for optimization of complex biological circuits across time scales.* by Charlotte Merzbacher, Oisin Mac Aodha, and Diego Oyarzún (2023).

## Requirements

This code is written in Python 3.x and uses the following packages:

- Pandas
- Hyperopt
- Numpy
- Scikits.odes
- pyDOE

For replicating the benchmarks, you will need:

- Scipy
- geneticalgorithm
- Time

And for replicating figures, you will need:

- Matplotlib
- Seaborn

## Supporting Code
1. **models.py** Includes the methods establishing all models, as well as helper functions needed for model computation.

## Analysis Notebooks
1. **toy_model_sim.ipynb** Runs a sample simulation and 100 background simulations. 
2. **benchmarking_methods.ipynb** Runs benchmarking methods (random sampling, grid search, gradient-based optimization, genetic algorithm) on toy model.
3. **glucaric_acid_sim.ipynb** Runs the glucaric acid model and kinetic perturbation experiments.
4. **fatty_acid_sim.ipynb** Runs the fatty acid model and objective function experiments.
5. **p_aminostyrene_sim.ipynb** Runs the p-aminostyrene model and chemical perturbation experiments.
6. **model_tuning_supp.ipynb** Runs model tuning and timing experiments.

## Visualization Notebooks
1. **toy_model_viz.ipynb** Creates contour plots of landscapes for Figure 1 and benchmarking visualizations.
2. **glucaric_acid_viz.ipynb** Plots sample run, perturbation experiment results, and dose-response curve plots (Figure 2).
3. **fatty_acid_viz.ipynb** Plots sample run and objective function Pareto curves (Figure 3).
4. **p_aminostyrene_viz.ipynb** Plots sample run and chemical robustness experiment results (Figure 4).
5. **supplementary_viz.ipynb** Creates all supplementary figures.

## Data Files
### Bayesian optimization for joint optimization of circuit architecture and parameters 
1. **toy_model_sample_run.csv** Contains the results of 1 500-iteration sample run of the toy model
2. **toy_model_background_bayesopt.csv** Contains the results of 100 500-iteration runs of the toy model for benchmarking.
3. **toy_model_landscapes_grid_search.csv** Contains the results of a 10x10 grid search of each of the architectures for contour landscape construction.
4. **toy_model_random_sampling_1000samples.csv** Triplicate results of 1000 random samples from parameter space (across all 4 architectures)
5. **toy_model_gradient_based.csv** 100 replicates of gradient-based solver for each of the 4 architectures
6. **toy_model_genetic_algorithm.csv** Triplicate results of genetic algorithm.

### Robustness of control circuits to uncertainty in enzyme kinetic parameters
1. **glucaric_acid_sample_run.csv** Contains the results of 1 1000-iteration sample run of the glucaric acid model
2. **glucaric_acid_background.csv** Contains the results of 100 1000-iteration runs of the glucaric acid model
3. **glucaric_acid_background_singlearch.csv** Contains the results of 100 1000-iteration runs of the glucaric acid model, with its architecture space restricted to a single control architecture at a time
4. **glucaric_acid_kinetic_perturbation.csv** [Large file, not on GitHub] Kinetic perturbation results across four architectures, 100 perturbations.

### Consistency of optima across flat objective function landscapes
1. **fatty_acid_sample_run_production_burden.csv** Contains the results of 1 1000-iteration sample run of the fatty acid model with the production-burden objective function
2. **fatty_acid_tradeoff_curve_speed_accuracy.csv** Contains the optimal results for scanning alpha values for the speed-accuracy objective function.


### Scalability of method to high-complexity large systems
1. **p_aminostyrene_sample_run.csv** Contains the results of 1 1000-iteration sample run of the p-aminostyrene model
2. **p_aminostyrene_background.csv** Contains the results of 100 1000-iteration runs of the p-aminostyrene model
3. **p_aminostyrene_chemical_robustness.csv** Chemical toxicity factor perturbation results.

## Figure Files
### Bayesian optimization for joint optimization of circuit architecture and parameters 
1. **toy_model_single_landscape_no_control.png** Contour plot of toy model landscape
2. **toy_model_single_landscape_upstream_repression.png** Contour plot of toy model landscape
3. **toy_model_single_landscape_downstream_activation.png** Contour plot of toy model landscape
4. **toy_model_single_landscape_dual_control.png** Contour plot of toy model landscape
5. **toy_model_benchmarking.png** Benchmarking bar plot.

### Robustness of control circuits to uncertainty in enzyme kinetic parameters
1. **glucaric_acid_sample_run.png** Sample run of glucaric acid model
2. **glucaric_acid_objective_fn_strip.png** Strip plot of perturbed system objective functions compared to background.
3. **glucaric_acid_optimal_arch_stacked_bar.png** Stacked bar of optimal architectures.
4. **glucaric_acid_dose_response_curves.png** Mean dose-response curves for dual control circuit.
5. **glucaric_acid_parameter_distributions.png** Parameter distributions across optimal circuit results.

### Consistency of optima across flat objective function landscapes
1. **fatty_acid_sample_run.png** Sample run of fatty acid model
2. **fatty_acid_optimality_curve.png** Tradeoff optimality curve based on speed-accuracy objective function
3. **fatty_acid_trajectories.png** Sample FFA trajectories from points on optimality curve

### Scalability of method to high-complexity large systems
1. **p_aminostyrene_sample_run.png** Sample run of p-aminostyrene model
2. **p_aminostyrene_objective_fn_strip.png** Strip plot of perturbed system objective functions compared to background.
3. **p_aminostyrene_parameter_pca.png** Scatter plot of PCA values for perturbed vs. background systems.

### Supplementary Figures
1. **supp_fatty_acid_objective_fn_strip.png** Strip plot of objective functions achieved by circuit, fatty acid model.
2. **supp_fatty_acid_coeff_var_barplot.png** Bar plot of coefficient of variation by circuit, fatty acid model.
3.**supp_model_hyperparameter_tuning.png** Results of manually tuning hyperparameter values
4. **supp_model_timing_barplot.png** Timing results of all models.

## Citation

If you use these methods and this code in your own research, please cite our paper:

*A machine learning approach for optimization of complex biological circuits across time scales.* by Charlotte Merzbacher, Oisin Mac Aodha, and Diego Oyarzún (2023).