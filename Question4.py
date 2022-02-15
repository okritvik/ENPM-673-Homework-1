"""
@author: Kumara Ritvik Oruganti
"""
import numpy as np

def get_s_matrix(sq_eig,A):
    sq_eig = np.sort(sq_eig)
    sq_eig = sq_eig[::-1]
    size = A.shape
    S = np.zeros(size)
    for i in range(0,len(sq_eig)):
        S[i][i] = sq_eig[i]
    print("S Matrix")
    print(S)
    return S

def get_u_matrix(A):
    AAT = np.matmul(A,A.T)
    eig_values, U= np.linalg.eig(AAT)
    eig_indices = sorted(range(len(eig_values)), key=list(eig_values).__getitem__)
    eig_indices.reverse()
    # print("Self Dec")
    # print(eig_indices)
    # idx1 = np.flip(np.argsort(eig_values))
    # print("Sakshi Dec")
    # print(idx1)
    # eig_values = eig_values[idx1]
    # U = U[:, idx1]
    
    
    U = U[:,eig_indices]
    print("U Matrix")
    print(U)
    return U

def get_vt_matrix(A):
    ATA = np.matmul(A.T,A)
    eig_values, Vt= np.linalg.eig(ATA)
    # print(Vt)
    eig_indices_asc = sorted(range(len(eig_values)), key=list(eig_values).__getitem__)
    eig_indices_asc.reverse()
    # print("Self ind")
    # print(eig_indices_asc)
    Vt = Vt[:,eig_indices_asc]
    # idx1 = np.flip(np.argsort(eig_values))
    # print("Sakshi Dec")
    # print(idx1)
    # eig_values = eig_values[idx1]
    # Vt = Vt[:, idx1]
    print("Vt Matrix")
    print(Vt.T)
    return Vt.T

def compute_homography(Vt):
    v = Vt.T
    x = v[:,-1]
    h = []
    for i in x.tolist():
        h.append(i[0])
    H = np.matrix([(h[0],h[1],h[2]),
                (h[3],h[4],h[5]),
                (h[6],h[7],h[8])
               ])
    print("H Matrix")
    print(H)
    

def q4_svd():
    x = [5, 150, 150, 5]
    y = [5, 5, 150, 150, 150]
    xp = [100, 200, 220, 100]
    yp = [100, 80, 80, 200]
    A = np.matrix([(-x[0], -y[0], -1, 0, 0, 0, x[0]*xp[0], y[0]*xp[0], xp[0]),
                   (0, 0, 0, -x[0], -y[0], -1, x[0]*yp[0], y[0]*yp[0], yp[0]),
                   (-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]),
                   (0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]),
                   (-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]),
                   (0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]),
                   (-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]),
                   (0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3])
                   ])
    print("A Matrix")
    print(A)
    print(A.shape)
    # print("EIGEN VALS")
    # print(np.linalg.eigvals(A*A.T))
    eig_val, eig_vec = np.linalg.eig(np.matmul(A,A.T))
    # print("Eigen Values")
    # print(eig_val)
    
    s = get_s_matrix(np.sqrt(eig_val),A)
    u = get_u_matrix(A)
    Vt = get_vt_matrix(A)
    
    # U,S,V = np.linalg.svd(A)
    # print("Original U:")
    # print(U)
    # print("Self U:")
    # print(u)
    # print("Original V:")
    # print(V)
    # print("Self V:")
    # print(Vt)
    # print("Original S:")
    # print(S)
    # print("Self S:")
    # print(s)
    # print("CHecking if UUT is unitary")
    # print(np.matmul(u,u.T))
    # print("CHecking if VVT is unitary")
    # print(np.matmul(Vt,Vt.T))
    
    return u,s,Vt

def main():
    U,S,Vt = q4_svd()
    compute_homography(Vt)
    
if __name__ == "__main__":
    main()