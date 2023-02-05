# Plot the (railway network) graph that is used for the graph problems

This is a spinoff of the `graphs/landkarte.py` which focusses on readability and writes the output into a file.
It is best called with the following statement to crop the resulting images accordingly:
`python3 landkarte.py; convert -crop 850x1100+190+260 Map.png Map.png; convert -crop 850x1100+190+260 Route.png Route.png`

