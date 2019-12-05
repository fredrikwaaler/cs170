To use the algorithm, simply import "algorithm" from TSP.
This is the DTH-algorithm, it will work by itself.
However, you will need the following libraries to run it:
- tspy

This library is installable trough pip.
"pip install tspy"

All you have to do is provide the algorithm with a input-file and a output-file,
like this: algorithm(input_file, output_file).

The algorithm is assuming that the input_file is a valid input-file as described in the project specification.
When called, the algorithm will do it's magic and write it's suggested solution as an output-file to the provided
output-file-path provided as a parameter.

