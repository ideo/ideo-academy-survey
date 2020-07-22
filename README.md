# ideo-academy-survey
There's three pieces of work in this repo right now, all related to work done between late June and early July 2020 for the IDEO Teams / IDEO Academy initiative.

1. There's a regression analysis component related to a survey that the team conducted. This was a relatively short stream of work that exists solely within the `eda_and_regression` jupyter notebook.
1. There's work related to applying locality sensitive hashing to group crunchbase company descriptions with value propositions analysis. Most of the `.py` scripts are related to this
1. Finally, there's a `web-app` directory that contains the output of a Sigma JS export from Gephi. It's mostly associated with a special branch (`network-viz`) that's connected to [a Netlify app](https://teams-lsh-network.netlify.app/#).

# getting started
## python setup
1. Create a **python 3.6.10** virtual environment. Below is a pyenv example which requires both `pyenv` and the `pyenv-virtualenvwrapper` to be installed (see the **Installation** section of [this page](https://gist.github.com/eliangcs/43a51f5c95dd9b848ddc) for help).
    ```
    pyenv virtualenv 3.6.10 academy-survey
    pyenv activate academy-survey
    ```

1. Install requirements.
    ```
    pip install -r requirements.txt
    ```

## data access
1. Create a symlink to [the dropbox folder](https://www.dropbox.com/home/IDEO-Academy-Survey) that contains all of the data. (Make sure you also have access to this dropbox folder). 
   ```
   ln -s ~/Dropbox\ \(IDEO\)/IDEO-Academy-Survey/ dropbox
   ```

## Gephi configuration
Gephi can be installed [directly from here](https://gephi.org/) for free. In order to export your work in a version that plays nicely with Netlify, you'll also need to import the Sigma JS exporter plugin (instructions shown in the first part of [this video](https://www.youtube.com/watch?v=lXt0DbDTOT8)). 

# Gephi + web app instructions
Most of the work to configure how the network visualization appears is done within Gephi's GUI itself, so long as you've run `make_gephi_files.py` to create the necessary `nodes.csv` and `edges.csv` files. There are only a couple of things worth mentioning here.

## use the network-viz branch for deploying
As you work in gephi, you'll be exporting things inside the `web-app` directory in this repository, which will create all the necessary files for Netlify to render the web app. _However_, changes will only appear at the web if you **switch to the network-viz branch** and push changes to that remote. Netlify will only update when it detects changes on _that_ branch.

## gephi settings
For posterity's sake, here are the settings that I used within Gephi to create the graph as it stands right now

- Set the *node size* to go according to the graph's *degree*, with a minimum size of 10 and a maximum size of 100.
- Use Gephi to calculate the *modularity* of the graph, and then set *node color* to reflect the four modularity classes.
- Set the *edges* to be colored according to their *weight*
- Finally, for the *layout* of the graph, run the Force Atlas 2 algorithm. Keeping the default settings is essentially OK, but for the time being I set *gravity equal to zero* and *scaling equal to 7.5*.

