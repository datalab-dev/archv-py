# Archv
Archv is the python version of the archive-vision project:
https://datalab.ucdavis.edu/archv/

# Setup for development

## 1. Install Appropriate Python version 3.8+

I recommend using homebrew for this if on a mac.
That way you can use homebrew to install opencv.
Homebrew's opencv has the nonfree stuff (SURF)
whereas opencv-python doesn't come with nonfree stuff. 
Which means they arent dependencies for this project, and can't be used.
This is very inconveniant
Installing opencv will also install the cv2.so to the homebrew installed python
```
brew install python3
brew install opencv
```

Otherwise you need to install opencv from source and copy the `.so` to the appropriate place. 
See this for reference:
[https://docs.opencv.org/4.5.2/dd/dd5/tutorial_py_setup_in_fedora.html](https://docs.opencv.org/4.5.2/dd/dd5/tutorial_py_setup_in_fedora.html)

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





