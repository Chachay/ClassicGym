import sympy as sy
import numpy as np
from scipy.signal import cont2discrete

class CostModel(object):
    def __init__(self, NX=None, NU=None):
        assert NX != None
        assert NU != None

        self.NX = NX
        self.NU = NU

        self.Lqq, self.Luu, self.Luq, \
        self.Lq, self.Lu, self.L,\
        self.Vqq, self.Vq, self.V = self.get_lambdified()

    def gen_cost_sympy_function(self):
        """
            returns stage and terminal cost function in sympy
        """
        raise NotImplementedError

    def get_lambdified(self):
        q = sy.symbols('q:{0}'.format(self.NX))
        u = sy.symbols('u:{0}'.format(self.NU))

        L, V = self.gen_cost_sympy_function()

        Lq = L.jacobian(q)
        Lu = L.jacobian(u)

        Lqq = sy.derive_by_array(Lq, q)
        Luu = sy.derive_by_array(Lu, u)
        Luq = sy.derive_by_array(Lu, q)

        Vq = V.jacobian(q)
        Vqq = sy.derive_by_array(Vq, q)

        return (*[sy.lambdify([q,u], F, ["numpy"]) for F in [Lqq, Luu, Luq, Lq, Lu, L]], 
                *[sy.lambdify([q], F, ["numpy"]) for F in [Vqq, Vq, V]])

class quadraticCostModel(CostModel):
    def __init__(self, Q=None, R=None, q=None, r=None,
                       Q_term=None, q_term=None, 
                       x_ref=None, NX=None, NU=None):
        assert NX != None
        assert NU != None

        assert Q.ndim==2 and Q.shape[0]==NX and Q.shape[1]==NX
        assert q.ndim==1 and q.shape[0]==NX
        assert R.ndim==2 and R.shape[0]==NU and R.shape[1]==NU
        assert r.ndim==1 and r.shape[0]==NU
        assert Q_term.ndim==2 and Q_term.shape[0]==NX and Q_term.shape[1]==NX
        assert q_term.ndim==1 and q_term.shape[0]==NX

        self.NX = NX
        self.NU = NU

        self.Q = Q
        self.q = q
        self.R = R
        self.r = r

        self.Qf = Q_term
        self.qf = q_term

        if x_ref is None:
            self.x_ref = np.zeros(NX)
        else:
            self.x_ref = x_ref

        super().__init__(NX=NX, NU=NU)
    
    def gen_cost_sympy_function(self):
        q = sy.symbols('q:{0}'.format(self.NX))
        u = sy.symbols('u:{0}'.format(self.NU))

        q_vec = sy.Matrix([e-self.x_ref[i] for i,e in enumerate(q)])
        u_vec = sy.Matrix([_ for _ in u])

        Q_weight = sy.Matrix(self.Q)
        R_weight = sy.Matrix(self.R)
        q_weight = sy.Matrix(self.q)
        r_weight = sy.Matrix(self.r)

        Qf_weight = sy.Matrix(self.Qf)
        qf_weight = sy.Matrix(self.qf)

        L = q_vec.transpose()*Q_weight*q_vec + u_vec.transpose()*R_weight*u_vec\
            + q_weight.transpose()*q_vec + r_weight.transpose()*u_vec
        V = q_vec.transpose()*Qf_weight*q_vec + q_weight.transpose()*q_vec

        return L, V
