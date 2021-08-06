# Archv
Archv is the python version of the archive-vision project:
https://datalab.ucdavis.edu/archv/

# Setup for development

## 1. Install Pyenv

On MacOS:  
`brew install pyenv`

Otherwise follow instructions from here:  
[https://github.com/pyenv/pyenv-installer](https://github.com/pyenv/pyenv-installer)

## 2. Install Poetry

from [https://python-poetry.org/docs/](https://python-poetry.org/docs/):

run the install script:  
`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`

open shell and test with:  
`poetry -version`

You may need to add poetry to your PATH in your shell config file

## 3. Set project specific python version with pyenv to 3.8.10

Install 3.8.10 with pyenv:  
`pyenv install 3.8.10`

Set local version to 3.8.10:  
`pyenv local 3.8.10`

## 4. Install dependencies with poetry

run:  
`poetry install`

You should be good to go. 
You can now drop into the virtualenv by running poetry shell, 
or run an arbitrary command in the virtualenv without dropping into it by using
`poetry run <your_command>`, for example `poetry run python some_script.py`

Want to add dependencies?  
Simply run `poetry add <package_name>`.
Don't forget to commit the resulting changes to pyproject.toml and poetry.lock!





