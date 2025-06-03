### Solar REU 2025 ReadMe

In this REU project, we will try to predict conditions near the Earth using HUXt. We will use Parker Solar probe measurements as inner boundary conditions to initialize a solar wind model in HUXt. 
This github repo contains tutorial notebooks to download data, generate orbits, and run HUXt. 

#### Cloning this repo on your machine:
```
git clone https://github.com/Lyosef789/HUXt_REU_project.git
```

### Initializing Submodules 

You will be using a copy of the HUXt software as a submodule. After cloning this main repo you will need to additionally pull the HUXt software with:

```
git submodule update --init --recursive
```

#### Running HUXt
HUXt has dependencies which are specified in `requirements.txt` and `environment.yml`. Therefore, we use **conda** to create a virtual environment following the recommendations in the HUXt repo. From the root directory of 'HUXt_REU_project', run the following commands:

```
>>conda env create -f environment.yml
>>conda activate huxt
```

Now, you should be able to run everything in this repo!

#### Notebooks

To start you off, inside the `code` folder we've prepared some starter Jupyter notebooks to get you started. 

`code/data_and_orbits.ipynb`

and 

`code/HUXt_first_steps.ipynb`

#### Lidiya has some reference code that may be helpful if you get stuck!

For alternative ways of downloading and plotting the data and orbits in its original format : 
`code/orbitals_download/Orbital_plotting.ipynb`

For HUXt usage
`code/HUXt_PSP_tutorial.ipynb`

You'll also find some useful multipurpose code in `/code/orbitals_download/helpers.py`
