# Code Base of Category Classification With Animal Image Data
This is a project by Chudan Ivy Liu and Xinyu Yang. All code was written in anaconda python 3.5.3.

# Install Required Dependencies
  1. Install package `openCV`
     * Run with command `pip install opencv-python`
  2. Install package `numpy`
     * Run with commcand `pip install numpy`
  3. Install package `matplotlib`
     * Run with command: `pip install matplotlib`
  4. Install package `scipy`
     * Run with command: `sudo pip install scipy -U`
  5. Install package `sklearn`
     * Run with command: `pip install -U scikit-learn`

# Overview of Design Decisions

    * `main.py`:  
       is the executable of interactive classification by training and saving the
       model to local disk
    * `data_processing.py`:
       loads and processes data, extracts features from instances_train2014 json file
    * `training.py`:
       trains data using four different algorithms
    * `predict.py`:
       predicts the test image using previously saved model (demo)
