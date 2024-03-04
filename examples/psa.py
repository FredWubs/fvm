import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import schur, rsf2csf, norm, LinAlgError, eig

# A=np.random.rand(10,10)
def psa(A,fname):
    # Set up grid for contour plot:
    npts = 50  # <- ALTER GRID RESOLUTION
    s = norm(A, 1)
    xmin, xmax, ymin, ymax = -s, s, -s, s  # <- ALTER AXES
    x = np.linspace(xmin, xmax, npts)
    y = np.linspace(ymin, ymax, npts)
    xx, yy = np.meshgrid(x, y)
    zz = xx + 1j * yy

    # Compute Schur form and plot eigenvalues:
    T, U = schur(A)
    if np.isrealobj(A):
       T, U = rsf2csf(T, U)
    T = np.triu(T)
    eigA = np.diag(T)
    plt.plot(np.real(eigA), np.imag(eigA), '.', markersize=15)
    plt.axis((xmin, xmax, ymin, ymax))
    #print(xmin,xmax,ymin,ymax)
    plt.axis('equal')
    plt.grid(True)
    plt.draw()

    # Reorder Schur decomposition and compress to interesting subspace:
    select = np.where(np.real(eigA) > -250)[0]  # <- ALTER SUBSPACE SELECTION
    n = len(select)
    for i in range(n):
        for k in range(select[i] - 1, i - 1, -1):
            G = np.array([[T[k, k+1], T[k, k] - T[k+1, k+1]]])
            G, _ = np.linalg.qr(G.T)
            J = slice(k, k+2)
            T[:, J] = T[:, J] @ G
            T[J, :] = G.T @ T[J, :]
    T = np.triu(T[:n, :n])
    I = np.eye(n)

    # Compute resolvent norms by inverse Lanczos iteration and plot contours:
    sigmin = np.inf * np.ones((len(y), len(x)))
    for i in range(len(y)):
        if np.isrealobj(A) and (ymax == -ymin) and (i > len(y) // 2):
            sigmin[i, :] = sigmin[len(y) - i, :]
        else:
            for j in range(len(x)):
                z = zz[i, j]
                T1 = z * I - T
                T2 = T1.conj().T
                if np.real(z) < 100:  # <- ALTER GRID POINT SELECTION
                    sigold = 0
                    qold = np.zeros(n)
                    beta = 0
                    H = np.zeros((100, 100))
                    q = np.random.randn(n) + 1j * np.random.randn(n)
                    q /= norm(q)
                    for k in range(99):
                        try:
                            v = np.linalg.solve(T1, np.linalg.solve(T2, q)) - beta * qold
                        except LinAlgError:
                            sig = np.inf
                            break
                        alpha = np.real(q.conj().T @ v)
                        v -= alpha * q
                        beta = norm(v)
                        qold = q
                        q = v / beta
                        H[k+1, k] = beta
                        H[k, k+1] = beta
                        H[k, k] = alpha
                        sig = np.max(np.linalg.eigvalsh(H[:k+1, :k+1]))
                        if np.abs(sigold / sig - 1) < 0.001 or (sig < 3 and k > 2):
                            break
                        sigold = sig
                    sigmin[i, j] = 1 / np.sqrt(sig)
        print(f'finished line {i+1} out of {len(y)}')

    # Plot contours
    fld=np.log10(sigmin + 1e-20)
    maxfld=round(np.max(fld))
    minfld=round(np.min(fld))
    levels=minfld+(maxfld-minfld)*np.arange(0, 1.1 ,0.1)
    plt.contour(x, y, fld , levels=minfld+(maxfld-minfld)*np.arange(0, 1.1 ,0.1))
    #plt.show()
    plt.savefig(fname)
    plt.clf()
 
    return 

