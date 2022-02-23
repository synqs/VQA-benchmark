# VQA-benchmark

This tool let's you run variational quantum algorithms (VQAs) to solve optimization problems.

## Quickstart

Edit and run `./main.py` to use the program. In this file, you make a choice of `options`. Then run

	single_run(options)

to simulate the chosen VQA up to a depth of p<sub>max</sub>` = options['pmax']`.

To compare different settings, use something like

	vary('qAlgorithm', all_options, options)

which will run _all_ possible `options['qAlgorithm']` and visualize them in a combined plot.
If you want to vary several parameters at once, this is the way to go:

	vary(('qAlgorithm', 'size'), all_options, options, how_many=2)

