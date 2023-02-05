# VQA-benchmark

This tool let's you run variational quantum algorithms (VQAs) to solve optimization problems.

## Installation

To create the necessary folders, run `python3 install.py`.
On a linux machine, the scripts should run by themselves. On Windows I did not test yet. It may however run with `python3.exe file.py`.

## Quickstart

Edit and run `./main.py` to use the program. In this file, you make a choice of `options`. Then run

	single_run(options)

to simulate the chosen VQA up to a depth of p<sub>max</sub> = `options['pmax']`.

To compare different settings, use something like

	vary('qAlgorithm', all_options, options)

which will run _all_ possible `options['qAlgorithm']` and visualize them in a combined plot.
If you want to vary several parameters at once, this is the way to go:

	vary(('qAlgorithm', 'size'), all_options, options, how_many=2)


## Choices within the framework

The tool is a modular system which allows choosing several parameters:
- Graph
    * Tiny  example (2 nodes, mostly for basic tests and debugging)
	* Small example (4 nodes)
	* German railway network (simplified, 13 nodes)
- Problem
	* Maximum-cut problem (MCP)
	* Travelling salesman problem (TSP)
	<!-- * Transverse field ising model (trafim) -->
- Quantum information unit
	* Qubit
	* Qudit (nothing implemented yet)
- Algorithm / circuit
	* VQE with sequential correlation (in multiple flavours, e.g. depending on correlating gate CNOT, CZ, RXX, RZZ, CPhase)
	* VQE with all-pair   correlation (again in multiple flavours)
	* QAOA
	<!-- * Grover (nothing implemented yet. What is that actually? does it help?) -->
- Hardware
	* Classical simulation
	* IBM quantum machine (not tested yet)
- Classical optimizer
	* Powell
	* Something with gradients (possible?)
	* Something own
- Maximum circuit layer number of `p` (program tests all integers from 1 to p<sub>max</sub>)


## Visualization

Several options control which images are plotted presenting the measured data. The results are however always stored as raw data and can therefore be (re-)plotted later on.

### Circuit layouts

With `options['print_circuit_images']` activated, the program plots the circuit layouts of the used circuits. This is first done before the optimizing including the names of the optimizable variables and afterwards including the values leading to the optimal final result.

### State distributions

Each quantum computation consists of `options['shots']` shots, by default 1024, each measured as a single result. This distribution is then evaluated for its average energy which is then used as the (negative) objective function by a classical optimization algorithm.

After running, the classical optimizer returns the parameter set that provided the best result. This is then being run again to get an insight into the final distribution of observations (TODO and, in the case of a quantum simulation, the full wave function). This distribution is then stored and (with |options['print_histograms']| activated) displayed in a diagram.

### Result overview

The tool automatically runs the circuit in different depths `p` from 1 up to `options['pmax']`. If called via `vary()`, the final results are saved and alltogether plotted into one diagram per inspected quality variable. Possible variables are the rate of optimal results, the average energy, the quantum entropy (TODO) and the runtime.
Each combination of tested parameters `options` leads to one line in the plot, always showing the different depths at the `x`-axis.


