import scipy
import scipy.linalg

if __name__ == "__main__":
    M = scipy.array([[1, 0, -1,1], [0, 1, 2, 1], [2, 0, -1, 2], [2, 0, -5, 3]])
    T, Z = scipy.linalg.schur(M)

    print("Z = ", Z)
    print('T = ', T)
    print(Z.dot(T).dot(Z.T) - M)