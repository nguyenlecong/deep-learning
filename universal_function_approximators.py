import numpy as np
import matplotlib.pyplot as plt
 
'''
x -> 1-hidden layer of 6 neurons -> z 
'''

curve = lambda x: (x-1)*(x+1)**2
ReLU = lambda x: np.maximum(0,x)
# ==== 1-hidden layer ===========
# consisting of 6 neurons with ReLU activation function
n1 = lambda x: ReLU(-5*x-7.7)
n2 = lambda x: ReLU(-1.2*x-1.3)
n3 = lambda x: ReLU(1.2*x+1)
n4 = lambda x: ReLU(1.2*x-0.2)
n5 = lambda x: ReLU(2.0*x-1.1)
n6 = lambda x: ReLU(5*x-5)
# ==== output ==========
# consisting of a output neuron with linear activation function
Z = lambda x: -1.0*n1(x)-1.0*n2(x)-1.0*n3(x)+1.0*n4(x)+1.0*n5(x)+1.0*n6(x)

# plotting approximators
xx = np.linspace(-2,1.5,150)
yy1 = Z(xx)
yy2 = curve(xx) 

plt.title('Xấp xỉ hàm bậc ba liên tục bằng mạng nơ ron 1-lớp ẩn')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
plt.plot(xx,yy1,linewidth=1.5,color='r',label='xấp xỉ')
plt.plot(xx,yy2,linewidth=1.5,color='b',label='$(x-1)*(x+1)^2$')
plt.xlim(-2,1.5)
plt.ylim(-1.3,1.3)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.legend()
plt.show()


