# ideo-academy-survey
survey regression analysis

# getting started

## python setup
1. Create a **python 3.6.10** virtual environment. Below is a pyenv example which requires both `pyenv` and the `pyenv-virtualenvwrapper` to be installed (see the **Installation** section of [this page](https://gist.github.com/eliangcs/43a51f5c95dd9b848ddc) for help)
    ```
    pyenv virtualenv 3.6.10 academy-survey
    pyenv activate academy-survey
    ```

1. Install requirements.
    ```
    pip install -r requirements.txt
    ```

## data access
1. Create a symlink to dropbox folder that contains all of the data. (Make sure you also have access to this dropbox folder). 
   ```
   ln -s ~/Dropbox\ \(IDEO\)/IDEO-Academy-Survey/ dropbox
   ```
