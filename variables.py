import numpy as np


m = 1
# limit of bound orbits, random choice? 
E = 0.973
L = 4.2*m
r = np.linspace(1,30,1000)
sch_Nitermax = 35
sch_epsilon = 10**-8
sch_start = 1
sch_dif = 10
sch_Ts = 1500
ini_tanv_sch = [0,9.05774475732293*m, 1/2*np.pi, 1.00000000000000e-12, 
               1.248725471366881, 0, 0, 0.0511928294380894/m]

# user can plot & choose which root finding method then enter them in?
sch_rper = 9.057744757322927
sch_rapo = 25.633769115462663

# kerr variables
a = 0.998
kerr_rp = 2.1752102281015393
kerr_ra = 6.852695179907905
kerr_T = 2000
mu = 1
E_k = 0.9
L_k = 2.
Q_k = 1.3
kerr_a = 0.998
kerr_Nitermax = 35
kerr_epsilon = 10**-8
kerr_x0 = 1.5 
