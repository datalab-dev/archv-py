# Archv
Archv is the python version of the archive-vision project:
https://datalab.ucdavis.edu/archv/

# Install

Follow instructions in setup.
Then run  
```
poetry build 
pip3 install dist/archv-0.1.0-py3-none-any.whl #change version appropriately
```
# Setup 

## 1. Install appropriate python version

Project requires python 3.8+ (see pyproject.toml)

I recommend using pyenv to manage python versions. 
However, if you will use homebrew to install opencv you *probably* need to use
homebrew to install an appropriate python version.

### 1. Using pyenv

Install pyenv: [https://github.com/pyenv/pyenv-installer](https://github.com/pyenv/pyenv-installer)
Install an appropriate python version:
```
pyenv install 3.8.10
pyenv local 3.8.10
```

### 2. Using homebrew
```
brew install python@3.9
```

## 2. Install Poetry

from [https://python-poetry.org/docs/](https://python-poetry.org/docs/):

run the install script:  
`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`

open shell and test with:  
`poetry -version`

You may need to add poetry to your PATH in your shell config file

## 3. Install dependencies with poetry

run:  
`poetry install`

You should be good to go. 
You can now drop into the virtualenv by running poetry shell, 
or run an arbitrary command in the virtualenv without dropping into it by using
`poetry run <your_command>`, for example `poetry run python some_script.py`

Want to add dependencies?  
Simply run `poetry add <package_name>`.
Don't forget to commit the resulting changes to pyproject.toml and poetry.lock!


## 4. Build Opencv and include cv2.so in site-packages

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

install opencv
```
brew uninstall opencv #in case there was a version built before installing brew python
brew install opencv 
```

copy the cv2.so to site-packages directory from above
```
cp /opt/homebrew/opt/python@3.9/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/*.so <site-packages-dir>
```
