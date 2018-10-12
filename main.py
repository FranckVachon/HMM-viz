import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

#matrix indicating how probably it is to change from hidden state ki hidden state kj, e.g. aij
transition_logprobs = np.log2(np.array([[.99,.01],
                                        [.01,.99]]))
transition_probs = np.array([[.95,.05],
                            [.05,.95]])

#our observation sequences, which we will use to try to deduce the likiest hidden state at each step
#to make things simple, here our sequence corresponds to the ROW number of the signal in the emission matrix
sequence_observations = np.array([1,1,1,1,0,0,0,0,0])

#the viterbi matrix - the one we build dynamically dims: [number_of_hidden_state, num_of_observations]
viterbi_matrix = np.zeros((transition_probs.shape[0], sequence_observations.shape[0]))

#the e matrix - each row is one of the possible signals we can emitt
#each column is a hidden state
#so each element is the probability of emitting a given signal for a given hidden state

emission_logprobs =np.log2( np.array([[.9,.1],
                                      [.1,.9]]))
emission_probs =np.array([[.8,.3],
                        [.2,.7]])

def dynamic_viterby_logprob(emission_probs, transition_probs, viter, obs):
    """using log prob to manage undeflow problems"""

    # init the first column of viterbi - this is just the probably of emitting the observed signal in each hidden state
    viter[:,0] = emission_probs[obs[0]]

    # for the dynamic part
    for idx,observation in enumerate(obs[0:-1]):

        #multiply transition_probs*viter[idx][:,None] and then takes the max of the each column of the resulting matrix.
        #this is equivalent to max(vk[i]*akl) in the book
        max_va = np.amax(transition_probs+(viter[:,idx][:,None]),axis=0)

        #now we have a [m,1] vector which we need to multiply by our emission probablity matrix.
        # Each element corresonds to one hidden state, and e = [n*m]
        viter[:, idx + 1] =(emission_probs+max_va[:,None])[observation,:]
    print(viter)
    return viter

def dynamic_viterby_normalize(emission_probs, transition_probs, viter, obs):
    """using normalisation on each column for underflow"""

    # init the first column of viterbi - this is just the probably of emitting the observed signal in each hidden state
    viter[:,0] = emission_probs[obs[0]]
    axis_sum = np.sum(viter[:,0])
    viter[:, 0] = viter[:,0]/axis_sum
    # for the dynamic part
    for idx,observation in enumerate(obs[0:-1]):

        #multiply transition_probs*viter[idx][:,None] and then takes the max of the each column of the resulting matrix.
        #this is equivalent to max(vk[i]*akl) in the book
        max_va = np.amax(transition_probs*(viter[:,idx][:,None]),axis=0)

        #now we have a [m,1] vector which we need to multiply by our emission probablity matrix.
        # Each element corresonds to one hidden state, and e = [n*m]
        viter[:, idx + 1] =(emission_probs*max_va[:,None])[observation,:]
        axis_sum = np.sum(viter[:, idx + 1])
        viter[:, idx + 1] /=axis_sum
    print(viter)
    return viter



def viterbi(image):
    """The emission matrix
    """
    row_labels = range(image.shape[0])
    col_labels = range(image.shape[1])
    plt.imshow(np.absolute(image), cmap='gray_r',)
    plt.xticks(range(image.shape[1]), col_labels)
    plt.yticks(range(image.shape[0]), row_labels)
    plt.show()




#img = dynamic_viterby_normalize(emission_probs,transition_probs,viterbi_matrix, sequence_observations)
#viterbi(img)

img_log = dynamic_viterby_logprob(emission_logprobs,transition_logprobs,viterbi_matrix, sequence_observations)
viterbi(img_log)


