# Archv
Archv is the python version of the archive-vision project:
https://datalab.ucdavis.edu/archv/

# Getting Started

Clone this repository
```
git clone https://github.com/datalab-dev/archv-py.git
```
Download anaconda
```
https://docs.anaconda.com/anaconda/install/
```

# Setup

## Install appropriate python version

Project requires python 3.8+ (see pyproject.toml)

I recommend using `pyenv` to manage python versions.

However, if you will use `homebrew` to install opencv you *probably* need to use `homebrew` to install an appropriate python version.

##  Using `homebrew` or `pyenv` to install appropriate version python

### homebrew
```
brew install python@3.9
```

### pyenv

Install pyenv: [https://github.com/pyenv/pyenv-installer](https://github.com/pyenv/pyenv-installer)

Install an appropriate python version:
```
pyenv install 3.8.10
pyenv local 3.8.10
```

## Install Poetry
from [https://python-poetry.org/docs/](https://python-poetry.org/docs/):

run the install script:  
`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`

open shell and test with:  
```
poetry --version
```


If the above gives you an error then you need to add to the config file to your PATH. Assuming that poetry is located in your $HOME directory, ie /User/bitnguyen/.poetry
```
export PATH="$HOME/.poetry/bin:$PATH"
```

Then rerun:
```
poetry --version
```

## Install dependencies with poetry

Inside **archv/** directory run:  
```
poetry install
```

You should be good to go.
You can now drop into the virtualenv by running poetry shell,
or run an arbitrary command in the virtualenv without dropping into it by using
`poetry run <your_command>`, for example `poetry run python some_script.py`

Want to add dependencies?  
Simply run `poetry add <package_name>`.
Don't forget to commit the resulting changes to pyproject.toml and poetry.lock!


## Build Opencv and include cv2.so in site-packages

First find the location of the site-packages directory for this project.
```
poetry run python -m site
```
find the one in the default pypoetry virtualenv location


### 1. From Source
Follow instructions from this page:
https://docs.opencv.org/4.5.2/dd/dd5/tutorial_py_setup_in_fedora.html

Then copy the cv2.so to site-packages directory from above
```
cp cv2.so <site-packages-dir>
```


### 2. With Homebrew

Create an environment in conda with python version 3.9:
```
conda create -n py39 python=3.9
```
Start the environment:
```
conda activate py39
```

Make sure you are in the environment:

`*` indicates which environment is active
```
conda info --envs
```

Your site packages should be in a similar directory like this:
```
/Users/bitnguyen/opt/anaconda3/envs/py39/python3.9/site-packages/
```

Install `opencv` with `brew`:
```
#in case there was a version built before installing brew python
brew uninstall opencv

brew install opencv
```

Copy the `cv2` directory from
> /usr/local/Cellar/opencv/4.5.3_2/lib/python3.9/site-packages/cv2

to your python site-packages:

```
cp /usr/local/Cellar/opencv/4.5.3_2/lib/python3.9/site-packages/cv2/ /Users/arthurkoehl/opt/anaconda3/envs/py39/lib/python3.9/site-packages/
```

# Run

Afterward, run:
```
poetry build

#change version appropriately
pip3 install dist/archv-0.1.0-py3-none-any.whl
```
