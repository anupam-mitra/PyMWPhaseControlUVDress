def calc_maxwellboltzmann (v, T, kB=1, m=1):
    p = sqrt(m/(2*pi*kB*T)) * exp(-m*v**2/(2*kB*T))
    return p
