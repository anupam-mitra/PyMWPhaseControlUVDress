# -*- coding: utf-8 -*-

import numpy as np
import pickle
import datetime, time, uuid, os
from numpy import pi

from unitarysequences import \
    RydbergAdiabaticPhasesTwoQubits, \
    TwoQubitSpin, \
    UncertainTwoQubitUnitaryQuantumControl, \
    ControlSequenceSearch


def main():
    np.random.seed(1)

    nsteps = 19
    distribwidth = 0.05
    outputdir = '.'

    time_stamp = time.time()
    time_string = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S')
    print("%s: Starting simulation for nsteps=%d, distribwidth=%g" %
          (time_string, nsteps, distribwidth))

    ra = RydbergAdiabaticPhasesTwoQubits(1, 0, pi/2)
    spin = TwoQubitSpin()
    target = spin.calc_yrotatetwist(0, pi)

    uqc = UncertainTwoQubitUnitaryQuantumControl(target, nsteps, ra, spin, distribwidth)
    search = ControlSequenceSearch(uqc, gtol=1e-6)
    angles_initial = pi*np.random.rand(2*nsteps)
    search.calc_optsequence(variables_initial=angles_initial)

    time_stamp = time.time()
    time_string = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d--%H-%M-%S')

    filename = time_string + '--' + str(uuid.uuid4()) + '.pkl'
    outputfile = open(os.path.join(outputdir, filename), 'wb')

    simulation_data = {
        'time_string': time_string,
        'uqc': uqc,
        'search': search,
    }

    pickle.dump(simulation_data, outputfile)
    outputfile.close()

    time_stamp = time.time()
    time_string = datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S')
    print("%s: Finished" % time_string)


if __name__ == '__main__':
    main()
