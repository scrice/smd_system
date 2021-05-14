import numpy as np

def append_systems(sys1,sys2):
    Abar = np.block([[sys1.A,np.zeros((2,2))], [np.outer(sys2.B,sys1.C), sys2.A]])
    Bbar = np.concatenate([sys1.B,np.zeros(2)])
    Cbar = np.concatenate([np.zeros(2),sys2.C])