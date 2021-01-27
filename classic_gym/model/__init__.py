import sympy as sy
import numpy as np
from scipy.signal import cont2discrete

class EnvModel(object):
    def __init__(self, NX=None, NU=None):
        assert NX != None
        assert NU != None

        self.NX = NX
        self.NU = NU

        self.Hqq, self.Huu, self.Huq, self.Jq, self.Ju, self.f = self.get_lambdified()

    def gen_rhe_sympy(self):
        raise NotImplementedError
    
    def get_lambdified(self):
        f = self.gen_rhe_sympy()
        q = sy.symbols('q:{0}'.format(self.NX))
        u = sy.symbols('u:{0}'.format(self.NU))
        
        Jq = f.jacobian(q)
        Ju = f.jacobian(u)

        Hqq = sy.derive_by_array(Jq, q)
        Huu = sy.derive_by_array(Ju, u)
        Huq = sy.derive_by_array(Ju, q)
        
        return (sy.lambdify([q,u], Hqq, ["numpy"]),
                sy.lambdify([q,u], Huu, ["numpy"]),
                sy.lambdify([q,u], Huq, ["numpy"]),
                sy.lambdify([q,u], Jq, ["numpy"]),
                sy.lambdify([q,u], Ju, ["numpy"]),
                sy.lambdify([q,u], f, ["numpy"]))
                
    def gen_dmodel(self, x, u, dT):
        f = self.f(x, u)
        A_c = self.Ju(x, u)
        B_c = self.Jq(x, u)
        
        x = x.reshape((-1,1))
        u = u.reshape((-1,1))

        g_c = f - A_c@x - B_c@u
        B = np.hstack((B_c, g_c))

        A_d, B_d, _, _, _ = cont2discrete((A_c, B, 0, 0), dT)
        g_d = B_d[:,self.NU]
        B_d = B_d[:,0:self.NU]

        return A_d, B_d, g_d

    def calc_RK4(self, x, u, dt):
        k1x = self.f(x, u).ravel()
        k2x = self.f(x+k1x*dt/2, u).ravel()
        k3x = self.f(x+k2x*dt/2, u).ravel()
        k4x = self.f(x+k3x*dt/2, u).ravel()

        x = x  + dt * (k1x/6 + k2x/3 + k3x/3 + k4x/6)
        return x

    def step_sim(self, x, u, dt, iterations=10):
        for _ in range(iterations):
            x = self.calc_RK4(x, u, dt/iterations)
        return x
