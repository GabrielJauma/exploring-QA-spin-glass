import networkx as nx
import numpy as np
import scipy.sparse as sparse
#import scipy.sparse.linalg 
import sys, slepc4py
from petsc4py import PETSc
from slepc4py import SLEPc


RGRAPH = 0
ANDERSON = 1
class RRG:
    def __init__(self, n, d = 3, t = 1,kind = RGRAPH):     
        self.n = n
        self.d = d
        self.t = t
        if kind == RGRAPH:
            self.n = n
        elif kind == ANDERSON:        
            self.L = n
            self.n = self.L**3
        self.kind = kind        
    #
    def nl(self):
#       G = nx.random_regular_graph(self.d, self.n, seed=None)  
        if self.kind == RGRAPH:
            print("rrg", self.kind)            
            G = nx.random_regular_graph(self.d, self.n) 
        if self.kind == ANDERSON:
            print("anderson", self.kind)
            ndim = self.n
            G = nx.grid_graph(dim = self.d*[self.L],periodic=True)
        return G
    def Hamiltonian(self, w, G):
        #random.seed(a=None)
        # diagonal elemnts
        di = np.random.uniform(-w/2, w/2, self.n)
        Hd = sparse.diags(di)
        # position of off diagonal elements
        ed = np.array(G.edges())
        col = ed[:,0]
        row = ed[:,1]
        # value of off diagonal elemnts
        noff = col.shape
        data = -self.t * np.ones(noff)        
        Mof = sparse.csr_matrix((data, (row, col)), shape=(self.n, self.n))
        Hoff = Mof + Mof.transpose()
        # Finally, the matrix in sparse form
        H = Hd + Hoff        
        return H;
    #
    def diagonaliza(self,Hp, nc, lo = False, slepc= True, mumps = False ):
        shape = Hp.shape
        n = shape[0]
        if slepc:
            #
            p1 = Hp.indptr
            p2 = Hp.indices
            p3 = Hp.data
            A = PETSc.Mat().createAIJ(size=Hp.shape,csr=(p1,p2,p3),comm=PETSc.COMM_SELF)
            E = SLEPc.EPS(); E.create()
            E.setOperators(A)
            E.setProblemType(SLEPc.EPS.ProblemType.HEP)
            E.setDimensions(nev=nc)
            E.setTolerances( tol=1e-14)
            E.setTarget(0)
            E.setWhichEigenpairs(slepc4py.SLEPc.EPS.Which.TARGET_MAGNITUDE)
            S = E.getST()
            S.setType(slepc4py.SLEPc.ST.Type.SINVERT)                                   
            if mumps:
                ksp = S.getKSP()
                ksp.setType('preonly')
                pc = ksp.getPC()
                pc.setType('lu')
                pc.setFactorSolverType('mumps') 
            E.solve()
            nconv = E.getConverged()
            print(nconv)
            if nconv >= nc:
                # Create the results vectors
                eie = np.zeros(nc)
                eiv = np.zeros((n,nc))                            
                vr, wr = A.getVecs()
                #
                for i in range(nc):
                    k = E.getEigenpair(i, vr)
                    eie[i] = k.real
                    eiv[:,i] = vr[:]
                    error = E.computeError(i)
        else:
            if lo:
                f = lambda y : Hp @ y
                H = sparse.linalg.LinearOperator(shape, matvec=f)
                lu = scipy.sparse.linalg.spilu(sparse.csc_matrix(Hp))
                g = lambda y : scipy.sparse.linalg.cg(H, y, tol = 0, x0 = lu.solve(y))[0]
                Oi = sparse.linalg.LinearOperator(shape, matvec=g)
                #start_time = time.time()
                # this is for scipy but using the iterative solver defined before
                eie,eiv = sparse.linalg.eigsh(H, nc, sigma=0, which='LM', OPinv = Oi, tol=1e-14)            
                end_time = time.time()
                #print("tiempo scipy",end_time-start_time)            
            else:
                H = Hp
                #start_time = time.time()
                eie,eiv = sparse.linalg.eigsh(Hp, nc,sigma=0, which='LM')            
                end_time = time.time()
                #print("timepo scipy",end_time-start_time)   
        return eie,eiv
    #
    def mstates(self, w, G, nc, de = None, wst = False, lo = False):
        Hp = self.Hamiltonian(w, G)
        eie, eiv = self.diagonaliza(Hp, nc , slepc = True )
        # write eigenvectors if needed:
        if wst:
            data_w = w * np.ones(eiv.shape[0] * eiv.shape[1])
            data_v = eiv.flatten()
            f=open(wst,'ab')
            np.savetxt(f,np.c_[data_w,data_v],fmt='%1.10f %1.9e')
            #np.savetxt(f, w,eiv,fmt='%1.9e')
            f.close()
        s = np.argsort(eie)       #devuelve eigenstates ordenados de menor a mayor
        eie1 = np.real(eie[s])
        eiv1 = np.real(eiv[:,s])
        H = None
        return eie1,eiv1;
    def one_sample(self, w, nc = 20, de = None, wst = False):

        G1 = self.nl()
        #G2 = self.nl()
        S1 = STAT (nc)
        nq = S1.nq()
        nsteps = np.size(w)
        a = np.zeros((nsteps, 2 * nq))

        for i in range(0, nsteps):
            e1, v1 = self.mstates(w[i], G1, nc, de = de, wst = wst)
            #e2, v2 = self.mstates(w[i], G2, nc)  
            xc = S1.ent(v1)
            a[i,0] = xc
            a[i,1] = xc**2
            xc = S1.I2( v1 )
            a[i,2] = xc
            a[i,3] = xc**2
            xc = S1.d0_max( v1)
            a[i,4] =  xc
            a[i,5] = xc**2
            #xc = S1.kls( v1 ) 
            #a[i,6] =  xc
            #a[i,7] = xc **2
            xc = S1.alpha0( v1 ) 
            a[i,6] =  xc
            a[i,7] = xc**2
            xc = S1.rs( e1 ) 
            a[i,8] =  xc
            a[i,9] = xc**2
        print(a[-1])
        return a;    
# 
class STAT:
    def __init__(self, nt, p = 0.05): 
        #self.n = n
        self.p = p
#         i0 = 0   #int(n/2-n*p)
#         i1 = nc  #int(n/2+n*p)
        self.i0 = 0
        self.i1 = nt
        self.nt = nt
    #
    def nq(self):
        return 5 # esto es el numero de magnitudes que pintamos
    # Las siguientes funciones son las que se calculan:
    def rs(self, E1):
        if self.nt > 2:
            x = sum( self.aux_r(np.take(E1, [j,j+1,j+2]))   for j in range(self.i0, self.i1-2))        
            xr =  x/(self.nt-2)
        elif self.nt == 2:
            xr =np.abs(E1[1]-E1[0])
        else:
            xr = E1[0]
        return xr        
    def ent(self,H1):
        x = sum( self.aux_S( H1[:,j]) for j in range(self.i0, self.i1))
        #x = x + sum( self.aux_S( H2[:,j]) for j in range(self.i0, self.i1))
        return x/(self.nt)
    def ent2(self,H1):
        x = sum( (self.aux_S( H1[:,j]))**2 for j in range(self.i0, self.i1))
        #x = x + sum( (self.aux_S( H2[:,j]))**2 for j in range(self.i0, self.i1))
        return x/(self.nt)      
    def I2(self,H1):
        x = sum( (self.aux_I2( H1[:,j])) for j in range(self.i0, self.i1))
        #x = x + sum( (self.aux_I2( H2[:,j])) for j in range(self.i0, self.i1))
        return x/(self.nt)      
    def x0(self,H1):
        x = sum( self.aux_x0( H1[:,j]) for j in range(self.i0, self.i1))
        #x = x + sum( self.aux_x0( H2[:,j]) for j in range(self.i0, self.i1))          
        return x/(self.nt);        
    #
    def kls(self,H1):   
        if self.nt > 1:
            x = sum( self.aux_kl( H1[:,j], H1[:,j+1]) for j in range(self.i0, self.i1-1))
            xr = x/(self.nt-1)
        else:
            xr =0
        #x = x + sum( self.aux_kl( H2[:,j], H2[:,j+1]) for j in range(self.i0, self.i1-1))
        return xr;
    #
    def alpha0(self,H1):
        x = sum( self.aux_x0(H1[:,j])for j in range(self.i0, self.i1))    
        return x
    #
    def d0_max(self,H1):
        return np.amax(H1**2)
    #
    #
    ###############################################################################
    # Aqui aparecen las funciones necesarias para calcular las funciones de arriba#
    ###############################################################################
    def aux_r(self, e):
        d = [ e[1]-e[0], e[2]-e[1] ]
        x = min(d[0],d[1])/max(d[0],d[1])
        return x;
    #
    def aux_kl(self, v1, v2):
        s1 = np.square(v1)
        s2 = np.square(v2)
        r1 = np.log(np.divide(s1,s2))
        x =np.dot(s1, r1)
#         x =  sum( v1[i]**2 * np.log( (v1[i]/v2[i])**2 )  for i in range(0, self.n ) )
        return x;    
    def aux_S(self, v1):
        ind = np.argwhere(np.abs(v1) > 1.0e-16)[:,0]
        s1 = np.square(v1[ind])
        r1 = np.log(s1)
        x = np.dot(s1, r1)
#         x =  sum( v1[i]**2 * np.log( v1[i]**2 )  for i in range(0, self.n ) )
        return x;     
    def aux_x0(self, v1):
        s1 = np.square(v1)
        r1 = np.log(s1)
        x =np.sum(r1)      
        return x
    def aux_I2(self, v1):
        s1 = np.power(v1,4)
        #r1 = s1
        x =np.sum(s1)            
        return x;                       #Library to run one sample with mpi scheme where:
def onesample_mpi(n, w, nvec = 20, de = None, wst = False):
    print("sample", de)
    red = RRG(n)
    a = red.one_sample(w, nc = nvec, de = de, wst = wst)
    #print(a)
    return a


