To use the algorithm, simply import "algorithm" from TSP.
This is the DTH-algorithm, it will work by itself.
However, you will need the following modules to run it:
- tspy

IMPORTANT: The tspy module while have some inherent errors.
There will be locations in the code where they have used indexing like [i,i].
You will have to manually change this to [i][i] for your local tspy module.
This can be done by using pycharm or pretty much any other high-level IDE, which will locate the line of the error for you.
This will give you the location of the wrong indexing [i,i], just click that location and you will be taken
to the code-file where you can change to [i][i].

This library is installable trough pip.
"pip install tspy"

All you have to do is provide the algorithm with a input-file and a output-file,
like this: algorithm(input_file, output_file).

The algorithm is assuming that the input_file is a valid input-file as described in the project specification.
When called, the algorithm will do it's magic and write it's suggested solution as an output-file to the provided
output-file-path provided as a parameter.

