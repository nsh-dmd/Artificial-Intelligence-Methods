import numpy as np

# Smoothing
def forward_backward( ev, prior, O, T ):
    t = len(ev)
    b = np.array( [1, 1] ) # representatiom of backward messeges, initially all are 1
    sv = np.array( [None]*(t+1) ) # vector of smoothed estimates [1, t]
    fv = np.array( [None]*(t+1) ) # vector of forward messeges[0, t]
    fv[0] = prior
    sv[0] = prior

    print("forward_msg( 0 ): ",fv[0])

#   Filtering process
    for i in range(1, t+1):
        fv[i] = forward( fv[i-1], ev[i-1], O, T )
        print( "forward_msg(", i, "): ",fv[i] )
    print("\n")

#   Smoothing process
    for j in range(t, -1, -1):
        sv[j] = normalize( fv[j] * b )
        print( "backward_msg(",j,"): ", b )
        b = backward( b, ev[j-1], O, T )
    return sv
    
# Filtering
def forward(fv, ev, O, T):
    obs = O[0] # Observation values for true evidence
    if not ev:
        obs = O[1]
    temp_msg = np.dot( obs, np.transpose(T) ) 
    for_msg = np.dot( temp_msg, fv )
    return normalize( for_msg )
    
# Updating 
def backward(b, ev, O, T,):
    obs = O[0] # Observation values for true evidence
    if not ev:
        obs = O[1]
    temp_msg = np.dot( T, obs )
    back_msg = np.dot( temp_msg, b )
    return back_msg

# returns normalized matrix
def normalize(matrix):
    return matrix / np.sum(matrix)
    print(type(np.sum(matrix)))

def main() :

    print( "\nVerifications for exercise 2\n****************************n" )
    
    # Transition model
    T = np.array([[0.7, 0.3],[0.3, 0.7]]) 

    # Observation model for True/False 
    O_true = np.array([[0.9, 0.0],[0.0, 0.2]]) #True
    O_false = np.array([[0.1, 0.0], [0.0, 0.8]]) #False
    O = [O_true, O_false]

    # The prior distribution on the initial state
    prior = np.array([0.5, 0.5]) 

    # evidence vector for t = 2
    ev_2 = [True, True]

    # evidence vector for t = 5
    ev_5 = [True, True, False, True, True]
    
    print("probability of rain at day 2:\n-------------------------------------------")
    smooth = forward_backward( ev_2, prior, O, T)
    
    print("\n ")
    for s in range(len(smooth)):
        print('smooth(',s,'):' , smooth[s])

    print("\nprobability of rain at day 5:\n--------------------------------------------")
    smooth = forward_backward( ev_5, prior, O, T)

    print("\n ")
    for s in range(len(smooth)):
        print( 'smooth(',s,'):' , smooth[s])


if __name__ == "__main__" :
    main()
