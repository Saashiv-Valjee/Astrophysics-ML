import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from pylab import rcParams
from AstroFunc import s_potential
from AstroFunc import effective_p
from AstroFunc import effectivep_fourtwo
from AstroFunc import sch_geod
from AstroFunc import sph2car
from AstroFunc import root_finder
from AstroFunc import kerr_geod
from AstroFunc import kerr_limits
from variables import m,E,L,r
from variables import sch_epsilon,sch_dif,sch_Nitermax,sch_start,sch_Ts, ini_tanv_sch,sch_rapo,sch_rper
from variables import kerr_a,kerr_epsilon,kerr_Nitermax,kerr_x0,kerr_ra,kerr_rp,kerr_T
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse


rcParams['figure.figsize'] = 5,5

e_dip = effectivep_fourtwo(E)
print(f'For specific angular momentum: {L} \nEffective potential dip at: {e_dip}')

V_r = s_potential(r,L)
plt.plot(r,V_r)
plt.ylim(-0.05,0.03)
plt.axhline(e_dip, color='k', linestyle='--', label="Limit of bound orbits")
plt.xlabel("r/m")
plt.ylabel("V(r)")
plt.grid()
plt.show()

# priming function for root finding
f = effective_p 

Nitermax = sch_Nitermax
epsilon = sch_epsilon

# starting guess - the variable that we change, 
# to find certain roots around specific x values. 
x0 = sch_start
x = x0
dif = sch_dif
n = 0

# find where V_r graph intersects e_dip
root,iter = root_finder(f,epsilon,x0,Nitermax,dif)
print(f"Root for Sch effective potential around {x0} is {root}\nIt took {iter} iterations")

T_s = sch_Ts

print('solving geodesics')
sol = solve_ivp(sch_geod, [0, T_s],ini_tanv_sch, t_eval=np.linspace(0, T_s, 1000))
r45, theta45, p45 = sol.y[1], sol.y[2], sol.y[3]

print('converting coordinates')
coords = sph2car(r45, theta45, p45)

print('Plotting Schwarzschild geodesics')
sch_rad = (sch_rapo - sch_rper) / m
m_aux = 1

ax1 = plt.subplot(1, 1, 1)

C = plt.Circle((0,0), 2*m_aux, color='k', label="Black hole")
Cperi = plt.Circle((0,0), sch_rper, color='y', alpha=0.1, label="Pericenter")
Capo = plt.Circle((0,0), sch_rapo, color='g', alpha=0.1, label="Apocenter")
ax1.add_patch(C)
ax1.add_patch(Cperi)
ax1.add_patch(Capo)
ax1.plot(coords[0], coords[1], color='r', label="RK45")
ax1.legend(loc='lower right')
plt.show()


print('Solving and plotting Kerr geodesics')

r_k = np.linspace(-1, 7, 1000)
rcParams['figure.figsize'] = 8, 4
plt.title("The effective potential of a Kerr Black hole")
plt.plot(r_k, kerr_limits(r_k), "r", label='R(r)')
plt.axhline(y=0, linestyle='--', color='k')
plt.xlabel("$r/m$")
plt.ylabel("$R(r)/m^4$")
plt.ylim(-5,5)

plt.legend()
plt.grid()

g = kerr_limits
Nitermax = kerr_Nitermax
epsilon = kerr_epsilon
x0 = kerr_x0
root,iter = root_finder(g,epsilon,x0,Nitermax,dif)
print(f"Root for Kerr effective potential around {x0} is {root}\nIt took {iter} iterations")

rp = 2.1752102281015393
ra = 6.852695179907905

a = kerr_a
rp = kerr_rp
ra = kerr_ra

T_k = kerr_T
ini_tanv_kerr = [0, (rp + ra)/2, np.pi/2, 0, 1.51876198038160, 0.187663512911857, 0.0559574180643787, 0.122475774691562]

ksolrk45 = solve_ivp(kerr_geod, [0, T_k],ini_tanv_kerr, t_eval=np.linspace(0, T_k, 2000))
ksolrk23 = solve_ivp(kerr_geod, [0, T_k],ini_tanv_kerr, t_eval=np.linspace(0, T_k, 2000), method='RK23')
ksolbdf = solve_ivp(kerr_geod, [0, T_k],ini_tanv_kerr, t_eval=np.linspace(0, T_k, 2000), method='BDF')
ksoldop = solve_ivp(kerr_geod, [0, T_k],ini_tanv_kerr, t_eval=np.linspace(0, T_k, 2000), method='DOP853')

kr45, ktheta45, kp45 = ksolrk45.y[1], ksolrk45.y[2], ksolrk45.y[3]
kr23, ktheta23, kp23 = ksolrk23.y[1], ksolrk23.y[2], ksolrk23.y[3]
krbdf, kthetabdf, kpbdf = ksolbdf.y[1], ksolbdf.y[2], ksolbdf.y[3]
krdop, kthetadop, kpdop = ksoldop.y[1], ksoldop.y[2], ksoldop.y[3]

kcoords45 = sph2car(kr45, ktheta45, kp45)
kcoords23 = sph2car(kr23, ktheta23, kp23)
kcoordsbdf = sph2car(krbdf, kthetabdf, kpbdf)
kcoordsdop = sph2car(krdop, kthetadop, kpdop)
traj_data = list(kcoords45),list(kcoords23),list(kcoordsbdf),list(kcoordsdop)


fig = plt.figure(figsize=(12, 8))  # Create a figure# Create data for the sphere

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)

# First subplot
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(x, y, z, color="black")
ax1.plot(kcoords45[0], kcoords45[1], kcoords45[2], alpha=0.75, linewidth=0.5, color='r', label="RK45")
ax1.legend(loc='lower right')
ax1.set_box_aspect([1, 1, 0.6])

# Second subplot
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot_surface(x, y, z, color="black")
ax2.plot(kcoords23[0], kcoords23[1], kcoords23[2], alpha=0.75, linewidth=0.5, color='g', label="RK23")
ax2.legend(loc='lower right')
ax2.set_box_aspect([1, 1, 0.6])

# Third subplot
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot_surface(x, y, z, color="black")
ax3.plot(kcoordsbdf[0], kcoordsbdf[1], kcoordsbdf[2], alpha=0.75, linewidth=0.5, color='b', label="BDF")
ax3.legend(loc='lower right')
ax3.set_box_aspect([1, 1, 0.6])

# Fourth subplot
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.plot_surface(x, y, z, color="black")
ax4.plot(kcoordsdop[0], kcoordsdop[1], kcoordsdop[2], alpha=0.75, linewidth=0.5, color='m', label="DOP")
ax4.legend(loc='lower right')
ax4.set_box_aspect([1, 1, 0.6])

plt.show()