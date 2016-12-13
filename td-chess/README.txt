td-chess

DISCLAIMER: Before you go through the setup of this project, realize that it is extremely slow and you will likely not
see any learning results.

To run this project, first go into the src/ directory and do
    $ pip install -r requirements.txt
That should give you all the dependencies, we're not sure if it properly installs tensorflow. If not, visit
https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html for more information on how to install tensorflow

Then check if you have a file named 'TrainingGames.pgn' in the the td-chess/data/ directory. If not,
you'll need the training games, which can be found at http://www.computerchess.org.uk/ccrl/4040/games.html.
Download the compressed PGN file of 'All games, without comments', unzip the file, rename it 'TrainingGames.pgn', and
place it in a directory called 'data' within the 'td-chess' directory. You may not have this file because it is very
large and GitHub's file size limit is 100MB.

To run the project, navigate to the src/ directory and type
    $ python main.py

This will first bootstrap the model using 1000 positions and the Stockfish evaluation function. Then we run the model
through STS, then train it using the sequential update function, then test it again.

If you'd like to run the batch update function instead of the sequential one, uncomment and comment the appropriate
lines in main.py. Take similar steps to test or not test in various places.

The PROB_THRESHOLD variable in td_model.py indicates the smallest value a node can have when searching before stopping
that particular path. 1.0 indicates no search is done; lower the threshold to search more nodes.

A variable of the same name exists in ai_agent.py, which controls the search depth the agent makes when choosing a move.

