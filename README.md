<p align="center">
<h1 align="center">Stability and invariance analysis of learned latent space dynamics</h1>
</p>
<p align="center">
</p>


## Installation

1. Clone the repository:
    ```
    $ git clone --
    ```
2. Before installing the required dependencies, you may want to create a virtual environment and activate it:
    ```
    $ virtualenv env
    $ source env/bin/activate
    ```
3. Install the dependencies necessary to run and test the code:
    ```
    $ pip install -r requirements.txt
    ```
4. Train the e2c:
    ```
    $ python train_pendulum.py -e 50
    ```
5. Encode images, compute the eigenvalues for A(z), clusterize the samples in the latent space, plot and save the clusters:
    ```
    $ python enc.py -g 1 -e 1 -p 1
    ```
6. Identify the matrixes (A,B) for the two clusters:
    ```
    $ python clusters_dyn.py -c 0
    ```
7. Decode the cluster centers and visualize the images:
    ```
    $ python dec.py
    ```
