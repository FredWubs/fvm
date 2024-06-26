# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:27:47 2023

Plot the field of values for the general eigenvalue problem
at specific rayleigh numbers

@author: tiesl
"""

import numpy
import scipy
import time
from scipy.sparse import identity
import psa
from fvm import plot_utils

from fvm import Continuation
from fvm import Interface

from jadapy import jdqz2
from jadapy.orthogonalization import orthonormalize

import matplotlib.pyplot as plt
from fvm.interface import JaDa



def rq(A, x):
    " Determines the rayleigh coefficient "
    return x.T.conj() @ A @ x / (x.T.conj() @ x)

    
def fv(B, nk=1, thmax=32, inverse=False):
    '''returns the eigenvalues and points of the fov as determined by the method
    written by Higham in matlab'''
    thmax=thmax-1
    
    iu = 1j
    n, p = numpy.shape(B)
    
    if n != p:
        print('Matrix must be square.')
        return
    
    f = numpy.zeros((2*thmax+1,nk), dtype='complex')
    z = numpy.zeros((2*thmax+1,1), dtype='complex')
    e, _ = numpy.linalg.eig(B)

    if numpy.all(B == B.T.conj()):
        f = [min(e), max(e)]
        
    elif numpy.all(B == -B.T.conj()):
        e = e.imag
        f = [min(e), max(e)]
        e = iu*e
        f = iu*f
        
    else:
        for m in numpy.arange(1,nk+1,1):
            ns = n-m
            A = B[0:ns+1, 0:ns+1]
            for i in numpy.arange(0, thmax+1, 1):
                th = i/thmax*numpy.pi
                Ath = numpy.exp(iu*th)*A
                H = 0.5*(Ath + Ath.T.conj())
                D, X = numpy.linalg.eig(H)
                k=numpy.argsort(D.real)
                z[i] = rq(A, X[:, k[0]])
                z[i+thmax] = rq(A, X[:, k[ns]])
            f[:,m-1] = z[:,0]
        
        #f[:,m] = f[:,0]
    
    if thmax==0:
        f = e
    
    if inverse:
        f = 1/f
    
    return f, e
        
def fovs(A, B):
    ''' plots the field of values of different combinations of the matrices A
    and B as given by Hochstenbach '''
    
    Ha = numpy.linalg.inv(B) @ A
    fa, e = fv(Ha)
#    print('fa')
#    for line in fa.real:
#      print('%e' % line)
#    for line in fa.imag:
#      print('%e' % line)

    #plt.axis('equal')
    plt.plot(e.real, e.imag, 'o')
#   plt.plot(fa.real, fa.imag, label='$W(B^{-1}A)$')
    plt.plot(fa.real, fa.imag)
    plt.axis('scaled')
    
#    Hb = A @ numpy.linalg.inv(B)
#    fb, _ = fv(Hb)
#    plt.plot(fb.real, fb.imag, ':', label='$W(AB^{-1})$')
#    
#    Hc = numpy.linalg.inv(A) @ B
#    fc, _ = fv(Hc, inverse=True)
#    plt.plot(fc.real, fc.imag, '--', label='$1/W(A^{-1}B)$')
#    
#    Hd = B @ numpy.linalg.inv(A)
#    fd, _ = fv(Hd, inverse=True)
#    plt.plot(fd.real, fd.imag, '-.', label='$1/W(BA^{-1})$')
#    print(fd)
    
#%% 

# initialize        
dim = 2
dof = 4
nx = 64
ny = nx
nz = 1
n = dof * nx * ny * nz

# Define the problem
parameters = {'Problem Type': 'Differentially Heated Cavity',
              # Problem parameters
              'Rayleigh Number': 1,
              'Prandtl Number': 1000.,
              'Reynolds Number': 1,
              # Problem size
              'X-max': 0.051,
              'Y-max': 1,
              # Set a maximum step size ds
              # 'Maximum Step Size': 1e8,
              # Set a smaller Newton tolerance
              'Newton Tolerance': 1e-14,
              'Optimal Newton Iterations': 4,
              'Delta': 1.0,
              # Give back extra output (this is also more expensive)
              'Verbose': False}

interface = Interface(parameters, nx, ny, nz, dim, dof)

continuation = Continuation(interface, parameters)

# Compute an initial guess
x0 = numpy.zeros(dof * nx * ny * nz)

previous_subspaces = None

# Rayleigh numbers to compute the fov at
data_points = [  2.88e9, 3.e12]
#data_points = [ 6.99e+07]
#data_points = [ 1.3e15, 3.e16, 3.e17, 3.e18  ]
#eigs = numpy.zeros([len(data_points), 10], dtype=numpy.complex128)

# continuate to the given rayleigh number, compute eigenvalues and return matrices
# to plot the fields of values with
AspRat=parameters.get('Y-max')/parameters.get('X-max')
mu=0.0
ds=data_points[0]/10
if True:
    target=3.e8
    restart=numpy.load(f'Results_Ra={target:.3}_A={AspRat:.3}.npz')
    x0=restart['x']
    A=restart['A']
    B=restart['B']
    q=restart['q']
    z=restart['z']
    mu=target
    ds=restart['ds']
    previous_subspaces = (q,)
    print(x0.size)
    print(A.size)
    print(q.size)
    print(len(q))
    print(len(q[0]))
for i, target in enumerate(data_points):
    x, mu, ds = continuation.continuation(x0, 'Rayleigh Number', mu, target, ds, last_ds=True)
    x0 = x

    # Compute the eigenvalues of the generalized eigenvalue problem
    jac_op = JaDa.Op(interface.jacobian(x))
    mass_op = JaDa.Op(interface.mass_matrix())
    # jada_interface = JaDa.Interface(interface, jac_op, mass_op, n, numpy.complex128)
    jada_interface = JaDa.BorderedInterface(interface, jac_op, mass_op, n, numpy.complex128)

    #For estimating the smallest tolerance possible
    # max absolute values of unknowns vector
    #maxu=numpy.linalg.norm(x[0:-1:4],ord=numpy.inf)
    #maxv=numpy.linalg.norm(x[1:-1:4],ord=numpy.inf)
    #maxp=numpy.linalg.norm(x[2:-1:4],ord=numpy.inf)
    #maxT=numpy.linalg.norm(x[3::4],ord=numpy.inf)
    #print('magnitudes u,v,p T',maxu, maxv, maxp,  maxT)
    # get the jacobian matrix explicitly.
    jac=jac_op @ identity(x.size)
    #print(jac)
    #detect the identity rows and romove them
    jd = scipy.sparse.diags(jac.diagonal())
    #print((numpy.abs(jac-jd)).sum(axis=1))
    #sum the absolute values of each row of the off-diagonal elements
    dumv=(numpy.abs(jac-jd)).sum(axis=1)
    for r in range(0,x.size):                   
      if dumv[r]==0.0:
         jac[r,r]=0
    #print((numpy.abs(jac[0::4,0::4])).max())
    #print(jac[0::4,0::4].multiply(jac[0::4,0::4].sign()))
    #print(jac[0::4,0::4].sabs().max())
    #print(scipy.linalg.norm(jac))
    #Compute 
    maxJ=numpy.zeros((4,4))
    maxx=numpy.zeros(4)
    maxJx=numpy.zeros((4,4))
    for r in range(0,4):
      maxx[r]=numpy.linalg.norm(x[r::4],ord=numpy.inf) 
    for r in range(0,4):
      for c in  range(0,4):
         maxJ[r,c]=(numpy.abs(jac[r::4,c::4])).max()
         maxJx[r,c]=maxJ[r,c]*maxx[c]
    print('maxJ \n',maxJ,'\n maxx \n',maxx,'\n maxJx \n',maxJx)
    #time.sleep()
         
    print(i)
    if i>0:
      print('check jdqz result before')
      print('orthogonality q \n',scipy.linalg.norm(q[:,:num].T.conj() @ q[:,:num]-numpy.identity(num)))
      print('orthogonality z \n', scipy.linalg.norm(z[:,:num].T.conj() @ z[:,:num]-numpy.identity(num)))
      print('approx A \n', scipy.linalg.norm(z[:,:num].T.conj() @ (jac_op @ q[:,:num])-Asav[:num,:num]))
      print('approx B \n', scipy.linalg.norm(z[:,:num].T.conj() @ (mass_op @ q[:,:num])-Bsav[:num,:num]))

    num=40
    #A, B, q, z = jdqz2.jdqz(jac_op, mass_op, num=num, tol=1e-12, subspace_dimensions=[30, 60], target=0,
    #                              interface=jada_interface, arithmetic='complex', prec=jada_interface.shifted_prec,
    #
    if True:
      A, B, q, z = jdqz2.jdqz(jac_op, mass_op, num=num, tol=1e-13, subspace_dimensions=[num, num+20], target=0.0+2.0j,
                                  interface=jada_interface, arithmetic='complex',
                                  initial_subspaces=previous_subspaces, return_matrix=True)
    else:
      A, B, q, z = jdqz2.jdqz(jac_op, mass_op, num=num, tol=1e-12, subspace_dimensions=[num, num+20], target=0,
                              interface=jada_interface, arithmetic='complex',
                              return_matrix=True)
    #previous_subspaces = (q, z)
    previous_subspaces = (q,)
    Asav=A.copy()
    Bsav=B.copy()
    print('norm inverse B',scipy.linalg.norm(numpy.linalg.inv(B[:num,:num])))
    values, eigenvecs=numpy.linalg.eig(numpy.linalg.solve(B[:num,:num],A[:num,:num]))
    print('kappa',scipy.linalg.norm(numpy.linalg.inv(eigenvecs[:num,:num]))*scipy.linalg.norm(eigenvecs[:num,:num]))
    
    
    print('check jdqz result')
    print('orthogonality q \n',scipy.linalg.norm(q[:,:num].T.conj() @ q[:,:num]-numpy.identity(num)))
    print('orthogonality z \n', scipy.linalg.norm(z[:,:num].T.conj() @ z[:,:num]-numpy.identity(num)))
    print('approx A \n', scipy.linalg.norm(z[:,:num].T.conj() @ (jac_op @ q[:,:num])-A[:num,:num]))
    print('approx B \n', scipy.linalg.norm(z[:,:num].T.conj() @ (mass_op @ q[:,:num])-B[:num,:num]))
    print('norm A \n', scipy.linalg.norm(A[:num,:num]))
    print('norm B \n', scipy.linalg.norm(B[:num,:num]))
    div_check=jac_op @ q
    indices = numpy.where((mass_op @  numpy.ones((n, 1))) == 0)[0]
    print('divergence check \n', scipy.linalg.norm(div_check[indices,:]))
    #time.sleep()
    q_M=q[:,:num].copy()
    #Switch to M orthogonal basis
    orthonormalize(q_M, M=mass_op)
    #print(q_M.T.conj()@(mass_op @ q[:,:num]))
    R_M=q_M.T.conj()@(mass_op@q[:,:num])
    print(numpy.matrix(R_M))
    print(scipy.linalg.norm(q[:,:num]-(q_M @ R_M)))
    print('before M innerproduct')
    print(numpy.diag(numpy.linalg.solve(B[:num,:num],A[:num,:num])))
#%% plot the fields of values
    
    A=numpy.linalg.solve(R_M.T.conj(),A[:num,:num].T.conj()).T.conj()
    B=numpy.linalg.solve(R_M.T.conj(),B[:num,:num].T.conj()).T.conj()
    print(numpy.diag(numpy.linalg.solve(B[:num,:num],A[:num,:num])))
    A=A[:num,:num]
    B=B[:num,:num]
    fovs(A, B)
    #plt.legend(loc='lower left')
    plt.title(f'Ra={target:.3}')
#%%    plt.show()
    #plt.axis('equal')
    plt.savefig(f'FOV_Ra={target:.3}_A={AspRat:.3}.eps')
    plt.clf()
    
#%% compute Binv A t
    ts = numpy.arange(0, 80000, 400)
    Binv = numpy.linalg.inv(B)
    R=Binv @ A
    psa.psa(R,f'PS_Ra={target:.3}_A={AspRat:.3}.eps')
    #print(numpy.matrix(R))
    vector=numpy.diag(R)
    print(vector)
    fil=open("Eigenvalues.txt","a")
    fil.write(f'PS_Ra={target:.3}_A={AspRat:.3}\n')
    #for line in vector:
    #numpy.savetxt(fil, vector, fmt='%2e\n')
    numpy.savetxt(fil, vector)
    fil.close()
#    time.sleep()
    fil=open("Matrices.txt","a")
    ys = []
    mat=numpy.matrix(R)
    for line in mat:
      numpy.savetxt(fil, line, fmt='%2e')
    fil.close()
    
    for t in ts:
        prod = Binv @ A * t
        exp = scipy.linalg.expm(numpy.array(prod,subok=True))
#%%        print(numpy.matrix(exp))
#%%        print(numpy.diag(exp)) 
        ys += [numpy.linalg.norm(exp,ord=1)]
#%%        print(ys)
#%%        input("Press Enter to continue...")
        
    # and plot
    if max(vector.real) <0:
       plt.plot(ts, ys)
       plt.yscale('log')
       plt.xlabel('t', fontsize=15)
#%%    plt.title(f'Ra={target:.3}')
#%%    plt.show()
       plt.ylabel('$||e^{B^{-1}At}||$', fontsize=15)
       plt.subplots_adjust(left=0.2, bottom=0.12)
       plt.savefig(f'Exp_Ra={target:.3}_A={AspRat:.3}.eps')
       plt.close()
#    plot_streamfunction(state, interface, axis=2, title='Streamfunction', *args, **kwargs)
    fig=plot_utils.plot_streamfunction(x, interface,show=False)
    fig.savefig(f'Sol_Ra={target:.3}_A={AspRat:.3}.eps')
    plt.close(fig='all')
    fig=plot_utils.plot_streamfunction(q[:,0], interface, show=False)
    fig.savefig(f'Ev_Ra={target:.3}_A={AspRat:.3}.eps')
#    plot_utils.plot_velocity_magnitude(q[:,0], interface)
#    plot_utils.plot_temperature(q[:,0], interface)

    plt.close(fig='all')
    fig=plot_utils.plot_temperature(x, interface,show=False)
    fig.savefig(f'SolT_Ra={target:.3}_A={AspRat:.3}.eps')
    plt.close(fig='all')
    fig=plot_utils.plot_temperature(q[:,0], interface, show=False)
    fig.savefig(f'EvT_Ra={target:.3}_A={AspRat:.3}.eps')
    plt.close(fig='all')
    
    fig=plot_utils.plot_pressure(x, interface,show=False)
    fig.savefig(f'SolP_Ra={target:.3}_A={AspRat:.3}.eps')
    plt.close(fig='all')
    fig=plot_utils.plot_pressure(q[:,0], interface, show=False)
    fig.savefig(f'EvP_Ra={target:.3}_A={AspRat:.3}.eps')
    plt.close(fig='all')

    if False:
      fil=open(f'Results_Ra={target:.3}_A={AspRat:.3}',"w")
      fil.write('state')
      fil.write(" ".join(map(str,x)))
      fil.write('A\n')
      mat=numpy.matrix(A)
      for line in mat:
        numpy.savetxt(fil, line)
      fil.write('B\n')
      mat=numpy.matrix(B)
      for line in mat:
        numpy.savetxt(fil, line)
      fil.write('q\n')
      mat=numpy.matrix(q)
      for line in mat:
        numpy.savetxt(fil, line)
      fil.write('z\n')
      mat=numpy.matrix(z)
      for line in mat:
        fil.write(" ".join(map(str,line))) 
      fil.close()

    numpy.savez(f'Results_Ra={target:.3}_A={AspRat:.3}',x=x,A=A,B=B,q=q,z=z,ds=ds)
