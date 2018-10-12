import matplotlib.pyplot as plt
import numpy as np


def dynamic_viterby_logprob(emission_probs, transition_probs, viter, obs):
    """using log prob to manage undeflow problems"""

    # init the first column of viterbi - this is just the probably of emitting the observed signal in each hidden state
    viter[:,0] = emission_probs[obs[0]]

    # for the dynamic part
    for idx,observation in enumerate(obs[1:-1]):

        #multiply transition_probs*viter[idx][:,None] and then takes the max of the each column of the resulting matrix.
        #this is equivalent to max(vk[i]*akl) in the book
        max_va = np.amax(transition_probs+(viter[:,idx][:,None]),axis=0)

        #now we have a [m,1] vector which we need to multiply by our emission probablity matrix.
        # Each element corresonds to one hidden state, and e = [n*m]
        viter[:, idx + 1] =(emission_probs+max_va[:,None])[observation,:]

        #now we have the value - need to tweak them a little so they show nicely on screen
        viter = np.absolute(viter)
        for col in range(0,viter.shape[1]):
            viter[:,col] = viter[:,col]/np.sum(viter[:,col])


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

def sanity_checks(emission_logprobs,transition_logprobs):
    sequence_observations_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    sequence_observations_1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    sequence_observations_2 = np.array([2, 2, 2, 2, 2, 2, 2, 2])
    l_test = [sequence_observations_0,sequence_observations_1,sequence_observations_2]
    for seq in l_test:
        init_viter = viterbi_matrix = np.zeros((transition_logprobs.shape[0], seq.shape[0]))
        img_log = dynamic_viterby_logprob(emission_logprobs, transition_logprobs, viterbi_matrix, seq)
        viterbi(img_log)

def parterns(emission_logprobs,transition_logprobs):
    sequence_observations_0 = np.array([0, 0, 0, 2, 2, 2, 2, 1])
    sequence_observations_1 = np.array([1, 2, 0, 1, 2, 0, 1,2])
    sequence_observations_2 = np.array([2, 2, 2, 1, 1, 1, 0, 0])
    l_test = [sequence_observations_0,sequence_observations_1,sequence_observations_2]
    for seq in l_test:
        init_viter = viterbi_matrix = np.zeros((transition_logprobs.shape[0], seq.shape[0]))
        img_log = dynamic_viterby_logprob(emission_logprobs, transition_logprobs, viterbi_matrix, seq)
        viterbi(img_log)

#the e matrix - each row is one of the possible signals we can emitt
#each column is a hidden state
#so each element is the probability of emitting a given signal for a given hidden state

emission_logprobs = np.log2(np.array([[.5,.25,.25],
                                        [.25,.5,.25],
                                        [.25, .25, .5]]))
#matrix indicating how probably it is to change from hidden state ki hidden state kj, e.g. aij

transition_logprobs = np.array([[.95,.05,.05],
                            [.05,.95,.05],
                             [.05, .05, .95]])

#our observation sequences, which we will use to try to deduce the likiest hidden state at each step
#to make things simple, here our sequence corresponds to the ROW number of the signal in the emission matrix


#the viterbi matrix - the one we build dynamically dims: [number_of_hidden_state, num_of_observations]
#viterbi_matrix = np.zeros((transition_logprobs.shape[0], sequence_observations.shape[0]))



#img_log = dynamic_viterby_logprob(emission_logprobs,transition_logprobs,viterbi_matrix, sequence_observations)
#viterbi(img_log)
parterns(emission_logprobs, transition_logprobs)

