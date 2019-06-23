# -*- coding: utf-8 -*-

import numpy as np
import pickle
import datetime, time, uuid, os
from numpy import pi

np.random.seed(1)

nsteps = 19
distribwidth = 0.05
outputdir= '.'

time_stamp = time.time()
time_string = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S')
print("%s: Starting simulation for nsteps=%d, distribwidth=%g" % \
      (time_string, nsteps, distribwidth))


from unitarysequences import \
    RydbergAdiabaticPhasesTwoQubits, \
    TwoQubitSpin, \
    TwoQubitUnitaryQuantumControl, \
    UncertainTwoQubitUnitaryQuantumControl, \
    ControlSequenceSearch

True
ra = RydbergAdiabaticPhasesTwoQubits(1, 0, pi/2)
spin = TwoQubitSpin()

target = spin.calc_zrotatetwist(-pi/2, pi) 
target = spin.calc_yrotatetwist(0, pi)

uqc = UncertainTwoQubitUnitaryQuantumControl(target, nsteps, ra, spin, distribwidth)
search = ControlSequenceSearch(uqc, gtol=1e-6)

#qc = TwoQubitUnitaryQuantumControl(target, nsteps, ra, spin)
#search = ControlSequenceSearch(qc)


# spin echo guess
#theta_initial = np.asarray([pi/2, pi, pi/2])
#phi_initial = np.asarray([0, 0, 0])
#angles_initial = np.reshape(np.asarray([theta_initial, phi_initial]), (2, nsteps))

# random guess
angles_initial = pi*np.random.rand(2*nsteps)


search.calc_optsequence(variables_initial=angles_initial)

time_stamp = time.time()
time_string = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d--%H-%M-%S')

filename = time_string + '--' + str(uuid.uuid4()) + '.pkl'
outputfile = open(os.path.join(outputdir, filename), 'wb')

simulation_data = {\
    'time_string': time_string, \
    'uqc': uqc, \
    'search' : search\
}

pickle.dump(simulation_data, outputfile)
outputfile.close()

# Finished
time_stamp = time.time()
time_string = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S')
print("%s: Finished" %  time_string)