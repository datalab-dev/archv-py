Python version of Archv
    Used for quicker development and testing, 
    not as replacement of cpp version

requires:
    python version 3.5
    opencv version 3.4

to install opencv for this project,
you can use the unofficial opencv package from pypi:
    pip install opencv-python 
    pip install opencv-contrib-python


To find similar images for all images within an image set.
First edit the settings.yml file
Second run process.py
Third run main.py

In the /data/ directory, store copies of settings.yml that you have used,
dictionaries created, and any other data you want to track


overview:
    main application (/archv):
        main.py
        process.py

    scripts (/bin):
        show.py
        draw_matches.py
        roi.py
        scan.py
        bag-of-words/
            bow.py
            dictionary.py
        

