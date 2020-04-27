# mFreeze

mFreeze is a computer vision package for analysing mouse behavioural experiments. It has a very small wrapper around another package: eztrack (https://github.com/DeniseCaiLab/ezTrack).


## Running Environment

Designed to be ran in jupyter notebooks. This is because it takes advantage of auto-updating cropping tools. 

## Installation

1. Create and activate a new python environment
2. Open a terminal
3. Type:
    ``` 
        $ git clone https://github.com/Ruairi-osul/mfreeze.git && cd mfreeze && pip install . 
    ```
4. To check installation, with your environment activated, type the following into the terminal:
    ``` 
        $ python
        >>> from mfreeze import FreezeProcessor
        >>>
    ```
5. If no errors occured, installation was successful.
    