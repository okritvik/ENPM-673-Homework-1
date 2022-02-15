"""
@author: Kumara Ritvik Oruganti
"""
import numpy as np
import os
import matplotlib.pyplot as plt
# import math
import random
# import time

def extract_data(ages,charges,csv_reader):
    for i in range(0,len(csv_reader)):
        ages.append(int(csv_reader[i][0]))
        charges.append(float(csv_reader[i][-1]))
        
        
def plot_raw_data(ages,charges,axes):
    axes.plot(ages,charges,'bo')
    
def plot_ls_curve(X,age,charges,ax):
    x = np.linspace(min(age),max(age),10000)
    y = X[0]*x + X[1]
    ax.plot(x,y,'r-')
    ax.plot(age,charges,'bo')
    ax.set_title("Least Squares")
    ax.set_xlabel("Age")
    ax.set_ylabel("Charges")
    
def least_squares(age,charges):
    sx2, sx, s, syx, sy = 0, 0, 0, 0, 0
    for i in range(0,len(age)):
        sx2 = sx2 + age[i]**2
        sx = sx + age[i]
        s = s + 1
        syx = syx + charges[i]*age[i]
        sy = sy + charges[i]
    A = np.array([(sx2,sx),(sx,s)])
    B = np.array([(syx,sy)]).transpose()
    A_inv = np.linalg.inv(A)
    X = np.matmul(A_inv,B)
    fig,axes = plt.subplots()
    plot_ls_curve(X,age,charges,axes)

def total_least_squares(ages,charges,axes):
    min_ages = min(ages)
    max_ages = max(ages)
    new_ages = []
    for i in range(0,len(ages)):
        new_ages.append((ages[i] - min_ages)/(max_ages-min_ages))
    min_charges = min(charges)
    max_charges = max(charges)
    new_charges = []
    for i in range(0,len(charges)):
        new_charges.append((charges[i] - min_charges)/(max_charges - min_charges))
    xbar = np.full(len(new_ages),np.mean(new_ages))
    # print(xbar)
    ybar = np.full(len(new_charges),np.mean(new_charges))
    # print(ybar)
    # print(ages)
    #print(np.full((1,len(ages)),xbar).shape)
    x_m_xbar = new_ages-xbar
    # print(x_m_xbar)
    # print("AGES 0")
    # print(ages[0])
    # print("XMBAR 0")
    # print(x_m_xbar[0])
    # print("AGES 0 - XMBAR 0")
    # print(ages[0]-xbar)
    y_m_ybar = new_charges-ybar
    # print("Converted list to array")
    # print(np.array(ages).reshape(1,-1).shape)
    U = np.vstack((np.array(x_m_xbar).reshape(1,-1),np.array(y_m_ybar).reshape(1,-1))).T
    print("U Shape")
    print(U.shape)
    UT = U.T
    print("UT Shape")
    print(UT.shape)
    UTU = np.matmul(UT,U)
    print(UTU.shape)
    eig_val, eig_vec = np.linalg.eig(UTU)
    # print(eig_val)
    min_eig_val_index = np.argmin(eig_val)
    print("Min eig val index")
    print(min_eig_val_index)
    corr_eig_vec = eig_vec[:, min_eig_val_index]
    print("EIG LEN")
    print(len(corr_eig_vec))
    # Checking if it the correct eig vec
    #check1 = np.matmul(UTU,corr_eig_vec)
    #check2 = np.multiply(corr_eig_vec,eig_val[min_eig_val_index])
    #print("CHECKING Started")
    #print(check1)
    #print(check2)
    #print("Checking ENDED")
    #print(min(eig_val))
    
    print(corr_eig_vec)
    # print(eig_val)
    # print(eig_vec)
    
    a = corr_eig_vec[0]
    b = corr_eig_vec[1]
    
    #print("Normalization Check")
    #print(math.sqrt(math.pow(a, 2)+math.pow(b,2)))
    
    d = a*xbar + b*ybar
    x = np.linspace(min(new_ages),max(new_ages),325)
    y = d/b-(a/b)*x
    
    # y = np.full((len(x),1),d)
    
    axes.plot(np.multiply(new_ages,max_ages),np.multiply(new_charges,max_charges),'bo')
    axes.plot(np.multiply(x,max_ages),np.multiply(y,max_charges),'g-')
    axes.set_title("Total Least Squares")
    axes.set_xlabel("Age")
    axes.set_ylabel("Charges")
    # u,s,v = np.linalg.svd(U)
    # print("Printing V")
    # print(v)
    # print("Printing sigma")
    # print(s)
    # small_sing_val_index = list(s).index(min(s))
    # vect = v[small_sing_val_index]
    # a = vect[0]
    # b = vect[1]
    # y = d/b- (a/b)*x

def ransac(ages,charges,e,p,t,s):
    min_ages = min(ages)
    max_ages = max(ages)
    new_ages = []
    for i in range(0,len(ages)):
        new_ages.append((ages[i] - min_ages)/(max_ages-min_ages))
    min_charges = min(charges)
    max_charges = max(charges)
    new_charges = []
    for i in range(0,len(charges)):
        new_charges.append((charges[i] - min_charges)/(max_charges - min_charges))
    best_a , best_b = 0, 0
    best_fit = 0
    N = np.inf
    samples_count = 0
    rand_points = [[],[]]
    d = 0
    while N>samples_count:
        rand_index = random.sample(range(0,len(new_ages)),s)
        rand_points[0] = [new_ages[rand_index[0]],new_ages[rand_index[1]]]
        rand_points[1] = [new_charges[rand_index[0]],new_charges[rand_index[1]]] 
        # print(rand_points)
        coeff_a , coeff_b, inliers, d_ret = ransac_tls(rand_points,new_ages,new_charges,t)
        print("NO of inliers = "+str(inliers))
        e = 1-(inliers/len(charges))
        N = int(np.log(1-p)/(np.log(1-np.power((1-e),s))))
        # print(" No. of Iterations = " + str(N))
        # print("CURR Sapmples count = "+ str(samples_count))
        # time.sleep(1)
        if(np.iscomplex(coeff_a) or np.iscomplex(coeff_b)):
            continue
        if inliers>best_fit:
            best_a = coeff_a
            best_b = coeff_b
            d = d_ret
        if(float(best_fit/N)>p):
            break
        samples_count = samples_count+1
        
    print("Iterations = "+str(samples_count))           
    print(best_a,best_b)
    fig,axes = plt.subplots()
    print(d)
    x = np.linspace(min(new_ages),max(new_ages),325)
    y = d/best_b-(best_a/best_b)*x
    axes.plot(np.multiply(new_ages,max_ages),np.multiply(new_charges,max_charges),'bo')
    axes.plot(np.multiply(x,max_ages),np.multiply(y,max_charges),'r-')
    # axes.plot(new_ages,new_charges,'bo')
    # axes.plot(x,y)
    axes.set_title("RANSAC")
    axes.set_xlabel("Age")
    axes.set_ylabel("Charges")
        
def ransac_tls(rand_points,new_ages,new_charges,t):
    x = rand_points[0]
    y = rand_points[1]
    xbar = np.mean(x)
    ybar = np.mean(y)
    x_m_xbar = x-xbar
    y_m_ybar = y-ybar
    U = np.vstack((np.array(x_m_xbar).reshape(1,-1),np.array(y_m_ybar).reshape(1,-1))).T
    # print("Shape of U")
    # print(U.shape)
    UT = U.T
    UTU = np.matmul(UT,U)
    eig_val, eig_vec = np.linalg.eig(UTU)
    min_eig_val_index = np.argmin(eig_val)
    corr_eig_vec = eig_vec[:, min_eig_val_index] 
    # print("IN RANSAC EIG")
    # print(corr_eig_vec)
    a = corr_eig_vec[0]
    b = corr_eig_vec[1] 
    d_inside = a*xbar + b*ybar
    inliers = 0
    for i in range(0,len(new_ages)):
        xt = (a*new_ages[i] + b*new_charges[i] - d_inside)
        if(np.abs(xt) <= t):
            inliers += 1
    return a,b,inliers,d_inside

def q3_a(ages,charges,axes):
    mean_ages = np.mean(ages)
    mean_charges = np.mean(charges)
    min_ages = min(ages)
    max_ages = max(ages)
    new_ages = []
    for i in range(0,len(ages)):
        new_ages.append((ages[i] - min_ages)/(max_ages-min_ages))
    min_charges = min(charges)
    max_charges = max(charges)
    new_charges = []
    for i in range(0,len(charges)):
        new_charges.append((charges[i] - min_charges)/(max_charges - min_charges))
    xbar = np.full(len(new_ages),np.mean(new_ages))
    # print(xbar)
    ybar = np.full(len(new_charges),np.mean(new_charges))
    # print(ybar)
    sxx = (np.sum(np.matmul((new_ages-xbar),np.transpose((new_ages-xbar)))))/len(new_ages)
    syy = (np.sum(np.matmul((new_charges-ybar),np.transpose((new_charges-xbar)))))/len(new_charges)
    sxy = (np.sum(np.matmul((new_ages-xbar),np.transpose((new_charges-xbar)))))/len(new_ages)
    syx = (np.sum(np.matmul((new_charges-ybar),np.transpose((new_ages-xbar)))))/len(new_charges)
    cov_matrix = np.array([(sxx,sxy),(syx,syy)])
    print("Covariance Matrix")
    print(cov_matrix)
    # print(np.linalg.det(cov_matrix))
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    # print(eigen_values)
    # print(eigen_vectors)
    # print("Original")
    v1 = eigen_vectors[:,0]
    v2 = eigen_vectors[:,1]
    # print(v1)
    # print(v2)
    #print("Check")
    #check_eigval,check_eigvec = np.linalg.eig(np.cov(ages,y=charges))
    # print(eigen_values)
    #print(check_eigvec[:,0])
    #print(check_eigvec[:,1])
    origin = [mean_ages,mean_charges]
    axes.quiver(*origin,*v1,color=['r'],scale=10)
    axes.quiver(*origin,*v2,color=['g'],scale=10)
    axes.set_title("Eigen Vectors of Co-variance Matrix")
    axes.set_xlabel("Age")
    axes.set_ylabel("Charges")

def q3_b(ages,charges):
    
    least_squares(ages,charges)
    
    fig,axes = plt.subplots()
    total_least_squares(ages,charges,axes)
    s=2
    e=0.1
    p = 0.99
    t = 0.05
    ransac(ages,charges,e,p,t,s)
    plt.show()
def main():
    path = os.getcwd()
    csv_reader = np.loadtxt(open(path+"/ENPM673_hw1_linear_regression_dataset - Sheet1.csv"),delimiter=",",skiprows=1,dtype='str')
    #print(csv_reader)
    ages = []
    charges = []
    extract_data(ages, charges, csv_reader)
    fig,axes = plt.subplots()
    plot_raw_data(ages, charges, axes)
    q3_a(ages,charges,axes)
    q3_b(ages,charges)
    

if __name__ == '__main__':
    main()