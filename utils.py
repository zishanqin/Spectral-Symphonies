import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, datetime


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Description of your program')
    
    # Input configuration
    parser.add_argument('--mode', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--load-npz-dir', type=str, help='Path to load train.npz and test.npz')
    parser.add_argument('--load-ckpt', type=str, help='Path to load checkpoint', default='')
    parser.add_argument('--ckpt-epoch', help='Number of epoch to select, used for resuming train and testing', default=None)
    
    # Training configuration
    parser.add_argument('--sample-size', type=int, help='Sampling size', default=200)
    parser.add_argument('--epoch-num', type=int, help='Number of epochs', default=20000)
    parser.add_argument('--class-num', type=int, help='Number of classes', default=25)
    parser.add_argument('--batch-size',type=int, help='Batch size', default=1024)
    
    # Feature configuration
    parser.add_argument('--L-reduced-dim', help='Reduced dimension for L (None for original dimension)', default=78)
    parser.add_argument('--F-reduced-dim', help='Reduced dimension for F (None for original dimension)', default=None)
    parser.add_argument('--selected-features', type=str,nargs="+",choices=['MFCC', 'DMFCC', 'FBC'], help='Names of selected feature', default=['MFCC', 'DMFCC', 'FBC'])
    
    # Parse arguments
    args = parser.parse_args()
    
    return args
    
def reduce_dimension(T, L_reduced, F_reduced):
    """
    Perform dimensionality reduction on feature tensor T.
    
    Args:
        T (ndarray): Feature tensor.
        L_reduced (int): Reduced dimension of L
        F_reduced (int): Reduced dimension of F
    """
    _, L, F, C = T.shape
    variance = 0
    T_reduced_list = [[] for _ in range(C)]
    if L_reduced is not None:
        for c in range(C):
            T_c = T[:,:,:,c]
            for f in range(F):
                T_f = T_c[:,:,f]
                pca = PCA(n_components=L_reduced)
                T_reduced = pca.fit_transform(T_f)
                variance += sum(pca.explained_variance_ratio_)
                T_reduced_list[c].append(T_reduced)
        T_final = np.moveaxis(np.array(T_reduced_list),[0,1,2,3], [3,2,0,1])
    else:
        for c in range(C):
            T_c = T[:,:,:,c]
            for l in range(L):
                T_l = T_c[:,l,:]
                pca = PCA(n_components=F_reduced)
                T_reduced = pca.fit_transform(T_l)
                variance += sum(pca.explained_variance_ratio_)
                T_reduced_list[c].append(T_reduced)
        T_final = np.moveaxis(np.array(T_reduced_list),[0,1,2,3],[3,1,0,2])
    return T_final, variance
    
def plot_history(network_history, feature_names):
    """
    Plot training loss and accuracy history.
    
    Args:
        network_history (tf.keras.callbacks.History): The CGAN network history.
        feature_names (str): Name of the channels.
    """
    now = str(datetime.now()).replace(" ","").replace(":","_").replace(".","").replace("-","_")
    fig = plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['train_loss'])
    plt.legend(['Training'])
    fig.savefig(f"./graphical_result/{feature_names}_training_loss_{now}.png")
    fig2 = plt.figure()
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(network_history.history['train_acc'])
    plt.legend(['Training'], loc='lower right')
    fig2.savefig(f"./graphical_result/{feature_names}_training_acc_{now}.png")
    

