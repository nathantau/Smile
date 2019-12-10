# Smile

======

Smile is a facial recognition software that is used to determine a person's key features. 
This project is heavily based off of using the concept of *localization*.

**Localization:** Determining the location of specific features of interest, given an image.

Smile is currently able to detect a person's eyes, nose, and mouth, with a high degree of accuracy.

# Machine Learning






# Sample Performance




# Directory

There are various `.py` files introduced in this project, and are separated to introduce modularity and clarity of code.

* `app.py` : This is the application's entrypoint. The user can choose to run `python app.py train` to train the neural network, or they can use `python app.py test` to test the network's capabilities on various test images.

* `img_reader.py` : This helper class is used to read images from a `.npz` file containing **numpy** representations of images. There are a total of 7049 images containerized in said `.npz` file.

* `csv_reader.py` : This helper class is used to read in landmark coordinates from a `.csv`, using the popular library **pandas**. There are 7049 rows of data corresponding to each image.

* `net.py` : This file contains a sandbox environment where I played around with different neural network architectures to achieve the best/highest accuracy results.

* `training.py` : This file contains helper methods used to train the neural network in a packaged manner.

* `prediction.py` : This file contains helper methods to use the neural network saved in an `.h5` format for predictive purposes.

