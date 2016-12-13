q-car

To run this project, make sure you are in the q-car/ directory and do
    $ pip install -r requirements.txt

You'll also need OpenAI's gym,
We'll provide installation instructions here, but if they are not sufficient, visit https://github.com/openai/gym

To get gym,
    $ git clone https://github.com/openai/gym.git
    $ cd gym
    $ pip install -e
    $ pip install -e .'[classic-control]'

To run the project:
    $ python qmodel.py

This will begin running an problem many times in a row, according to the experiment specifications in our writeup.
If you'd like to view the game, set RENDER in qmodel.py to True