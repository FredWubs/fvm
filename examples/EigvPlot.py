import ReadOutput
import matplotlib.pyplot as plt
flnm='Eigenv.out'
word='eigenvalue:'
evs=ReadOutput.ReadOutput(flnm,word)
#plt.axis('equal')
plt.plot(evs.real, evs.imag, 'o')
#plt.axis('scaled')
#plt.show()
#fig.savefig(f'Ev_Re={target:.3}_A={AspRat:.3}.eps')
plt.savefig('LDC_eigenv.eps')

