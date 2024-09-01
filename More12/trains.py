import matplotlib.pyplot as plt
import numpy as np

def fa1(x):  #- lapalcian u=-u^2   {on }[0,1], u(0)=0,u(1)=1.
    import numpy as np
    lamda=1
    p=2    
    N=len(x)
    h=1/(N-1)
    A=np.zeros((N,N))    
    #A=np.zeros((N,N))
    i,j = np.indices(A.shape)
    A[i==j] = 2
    A[i==j-1] = -1
    A[i==j+1] = -1
    A[0,1]=0
#    A[1,0]=0 
    A[-1,-2]=0
#    A[-2,-1]=0
    b=np.zeros((N,1))
#    b[-2,0:1]=1/h
    b[-1,0:1]=2/h
    A=A/h
    def f(x):
        #M=len(x)
        #g=np.zeros((M,1),dtype=complex)
        g=lamda*((x**p))
        g[0,0:1]=0
        g[-1,0:1]=0
        return g
    def F(x):       
        return A@x+h*f(x)-b
    def df(x):
        #g1=np.zeros((N,N),dtype=complex)
        g1=np.identity(len(x))*(lamda*(p*(x**(p-1))))
        return g1
    def dF(x):
        return A+h*df(x)
    return [F,dF]


class PowerSeries():
    r"""Power series.

    p(x) = \sum_{i=0}^{N-1} a_i x^i

    Args:
        N (int)
        M (float): `M` > 0. The coefficients a_i are randomly sampled from [-`M`, `M`].
    """

    def __init__(self, N=100, M=1):
        self.N = N
        self.M = M

    def random(self, size):
        return 2 * self.M * np.random.rand(size, self.N) - self.M

    def eval_one(self, feature, x):
        return np.dot(feature, x ** np.arange(self.N))

    def eval_batch(self, features, xs):
        mat = np.ones((self.N, len(xs)))
        for i in range(1, self.N):
            mat[i] = np.ravel(xs ** i)
        return np.dot(features, mat)

    def eval_batch1(self, features, xs):
        mat = np.ones((self.N, len(xs)))
        for i in range(0, self.N):
            if i<1:
                mat[i]=0
            else:
                mat[i] = np.ravel(i*xs ** (i-1))
        return np.dot(features, mat)
    
    def eval_batch11(self, features, xs):
        mat = np.ones((self.N, len(xs)))
        for i in range(0, self.N):
            if i<2:
                mat[i]=0
            else:
                mat[i] = np.ravel(i*(i-1)*xs ** (i-2))
        return np.dot(features, mat)
    





degree = 3
L=1
space = PowerSeries(N=degree + 1,M=L)

num_func=1200
num_eval_points=400
evaluation_points = np.linspace(0,1,num_eval_points, endpoint = True).reshape(num_eval_points,1)
func_feats = space.random(num_func)
#print(func_feats)
func_vals = space.eval_batch(func_feats, evaluation_points)
y=func_vals.copy()

yy=space.eval_batch11(func_feats, evaluation_points)


xx=-(3*np.ones((num_eval_points,1)))**2
[F,dF]=fa1(xx)
for i in range(50):
    if abs(F(xx)).max()<10**(-13):
        break
    xx=xx-np.linalg.inv(dF(xx))@F(xx)
#plt.plot(xx)
func_vals[:int(num_func/2),:]=func_vals[:int(num_func/2),:]+np.transpose(xx)



xx=-(3*np.zeros((num_eval_points,1)))**2
[F,dF]=fa1(xx)
for i in range(50):
    if abs(F(xx)).max()<10**(-13):
        break
    xx=xx-np.linalg.inv(dF(xx))@F(xx)
#plt.plot(xx)
func_vals[int(num_func/2):,:]=func_vals[int(num_func/2):,:]+np.transpose(xx)


#print(np.shape(xx))
#print(np.shape(func_vals))
#for i in range(499,520):
#    plt.plot(func_vals[i,:])
train_x = (func_vals, evaluation_points)
#print(yy)


x=func_vals[0].reshape((len(func_vals[0]),1)).copy()
[F,dF]=fa1(x)
#print(F(x))
#plt.plot(x)
#plt.show()
#plt.plot(F(x))
for ii in range(len(func_vals)):
    x=func_vals[ii].reshape((len(func_vals[ii]),1)).copy()
    for i in range(1):
        x=-np.linalg.inv(dF(x))@F(x)
        y[ii]=x.reshape(len(func_vals[ii]))

func_vals = np.array(func_vals, dtype=object)
evaluation_points= np.array(evaluation_points, dtype=object)
y = np.array(y, dtype=object)
yy = np.array(yy, dtype=object)
np.savez('100train.npz', x=func_vals,xx=evaluation_points, y=y,yy=yy)

