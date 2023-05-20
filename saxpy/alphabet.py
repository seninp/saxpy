"""Implements Alphabet cuts."""
import numpy as np
import scipy

def cuts_for_asize(a_size):
    """Generate a set of alphabet cuts for its size.

    Typically, we generate cuts in R as follows:
        get_cuts_for_num <- function(num) {
            cuts = c(-Inf)
            for (i in 1:(num-1)) {
                cuts = c(cuts, qnorm(i * 1/num))
                }
                cuts
        }

        get_cuts_for_num(3)

    Args:
        a_size (int): Number of sections to cut into 
                      (Must be between 2 and 26)

    Returns:
        numpy.array: Numpy array of the threshold of the cuts
    """    

    # Only limit too 26 sections as there are only 26 letters in the alphabet
    if a_size > 26 or a_size < 2:
        raise ValueError("'a_size must be between 2 and 26'") 

    break_points = [-np.inf]

    for i in range(1, a_size):
        break_points.append(scipy.stats.norm.ppf(i/a_size))
        
    return np.array(break_points)
