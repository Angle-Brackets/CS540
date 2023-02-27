#HW3 By Ankit Joju

from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    return x - np.mean(x, axis=0)

def get_covariance(dataset):
    const =  (1/(len(dataset) - 1)) 
    
    S = np.dot(np.transpose(dataset), dataset) * const
    return S

def get_eig(S, m):
    Lambda, U =  eigh(S, subset_by_index=[len(S)-m, len(S)-1])

    #The eigenpairs list is an array of tuples, where it's formatted as (eigenvalue, eigenvector), and sort only looks at the first value.
    eigenpairs = list()
    eigenvals = list()
    eigenvectors = list()
    for i in range(len(Lambda)):
        eigenpairs.append((Lambda[i], U[:, i]))

    eigenpairs = sorted(eigenpairs, reverse=True)
    
    #This loops through and adds each pair to the requisite array.
    for i in range(len(eigenpairs)):
        eigenvals.append(eigenpairs[i][0])
        eigenvectors.append(eigenpairs[i][1])
    
    #Take all of the eigenvalues and put them into a diagonal matrix, and put the vectors in a column vector matrix.
    Lambda = np.diag(eigenvals)
    U = np.column_stack(eigenvectors)

    return Lambda, U

def get_eig_prop(S, prop):
    eigens = eigh(S, eigvals_only=True)
    eigen_sum = np.sum(eigens)
    desired_eigenvalues = list()


    for eigen_value in eigens:
        variance = eigen_value / eigen_sum
        if variance > prop:
            desired_eigenvalues.append(eigen_value)
    
    desired_eigenvalues = sorted(desired_eigenvalues, reverse=True)
    Lambda = np.diag(desired_eigenvalues)
    
    ExtraLambda, U = eigh(S, subset_by_value=[desired_eigenvalues[-1] - 1, desired_eigenvalues[0] + 1])

    eigenpairs = list()
    eigenvectors = list()

    for i in range(len(Lambda)):
        eigenpairs.append((ExtraLambda[i], U[:, i]))
    eigenpairs = sorted(eigenpairs, reverse=True)

    for i in range(len(eigenpairs)):
        eigenvectors.append(eigenpairs[i][1])
    
    U = np.column_stack(eigenvectors)

    return Lambda, U

def project_image(image, U):
    alpha = 0
    cols = U.shape[1]
    
    for j in range(cols):
        alpha += np.dot(np.dot(np.transpose(U[:, j]), image), U[:, j])
    
    return alpha

def display_image(orig, proj):
    orig_tmp = np.transpose(np.reshape(orig, (32, 32)))
    proj_tmp = np.transpose(np.reshape(proj, (32, 32)))
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    c1 = ax1.imshow(orig_tmp, aspect="equal")
    c2 = ax2.imshow(proj_tmp, aspect="equal")

    ax1.set_title("Original")
    ax2.set_title("Projection")

    f.colorbar(c1, ax=ax1)
    f.colorbar(c2, ax=ax2)
    f.tight_layout(pad=2.0)

    plt.show()
x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
projection = project_image(x[0], U)
display_image(x[0], projection)