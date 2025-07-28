#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import symbols, Function, diff, simplify, solve, lambdify, Matrix
from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols
from sympy.printing.latex import latex

t = symbols('t')
q1, q2 = dynamicsymbols('q1 q2')
dq1 = diff(q1, t)
dq2 = diff(q2, t)
m, k, kc = symbols('m k kc') 

L = (1/2)*m*(dq1**2 + dq2**2) - (1/2)*k*(q1**2 + q2**2) - (1/2)*kc*(q1 - q2)**2

qs = [q1, q2]
lagrange_method = LagrangesMethod(L, qs)
eoms = lagrange_method.form_lagranges_equations()
eoms = simplify(Matrix(eoms))

from IPython.display import display, Math
for i, eq in enumerate(eoms):
    display(Math(f"Ecuación {i+1}: " + latex(eq) + " = 0"))

qdds = [diff(q1, t, t), diff(q2, t, t)]
sols = solve(eoms, qdds)

funcs_qdd = [lambdify((t, q1, dq1, q2, dq2, m, k, kc), sols[qdd], modules='numpy') for qdd in qdds]

def system(t_val, y, m_val=1, k_val=1, kc_val=0.5):
    q1_val, dq1_val, q2_val, dq2_val = y
    ddq1_val = funcs_qdd[0](t_val, q1_val, dq1_val, q2_val, dq2_val, m_val, k_val, kc_val)
    ddq2_val = funcs_qdd[1](t_val, q1_val, dq1_val, q2_val, dq2_val, m_val, k_val, kc_val)
    return [dq1_val, ddq1_val, dq2_val, ddq2_val]

y0 = [1, 0, -1, 0]
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)

sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

plt.figure(figsize=(10,5))
plt.plot(sol.t, sol.y[0], label='q₁(t)')
plt.plot(sol.t, sol.y[2], label='q₂(t)')
plt.xlabel('t')
plt.ylabel('Posición')
plt.title('Solución de sistema acoplado')
plt.legend()

plt.text(0.5, 0.9, r"$" + latex(eoms[0]) + r" = 0$", fontsize=14, transform=plt.gca().transAxes,
         horizontalalignment='center')

plt.grid(True)
plt.show()


#%%
from sympy.physics.mechanics import dynamicsymbols, LagrangesMethod
from sympy import symbols, diff, simplify

t = symbols('t')
q = dynamicsymbols('q')
qd = diff(q, t)
m, k = symbols('m k')

L = (1/10)*m*qd**2 - (1/2)*k*q**2

LM = LagrangesMethod(L, [q])

p_q = diff(L, qd)
print(f"Momento conjugado p_q = {p_q}")

dL_dq = diff(L, q)
print(f"Dependencia explícita de L respecto a q: {dL_dq}")

if dL_dq == 0:
    print("q es coordenada cíclica => p_q es constante (cantidad conservada)")
else:
    print("q NO es coordenada cíclica => No se conserva p_q")

