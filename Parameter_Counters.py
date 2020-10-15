import numpy as np


'''
==================
      LeNet-5           
==================
        filters=6             Avg. pool            filters=16
[32,32] ----------> [28,28,6] ---------> [14,14,6] ----------> [10,10,16]...
         k=[5,5]              k=[2,2]              k=[5,5]   
         s=1                  s=2                  s=1
         p=0                                       p=0 
         a='tanh'                                  a='tanh'
           Avg. pool           FC          FC         FC
[10,10,16] ---------> [5,5,16] --> [120,1] --> [84,1] --> softmax([10,1])    
            k=[2,2]  
            s=2
                        
'''
print('Counting # of parameters in the LeNet-5')
print('f=6,k=[5,5],s=1,p=0')
w1 = 6*5*5 + 6
print('w1=',w1)
print('f=10,k=[5,5],s=1,p=0')
w2 = 10*5*5 + 10
print('w2=',w2)
print('FC [5,5,16]x120')
w3 = 16*5*5*120 + 120
print('w3=',w3)
print('FC 120x84')
w4 = 120*84 + 84
print('w4=',w4)
print('FC 84x10')
w5 = 84*10 + 10
print('w5=',w5)
print('Total number of parameters:',w1+w2+w3+w4+w5)
'''
================
 Alex Net : Bai tap tren lop
================
             f=96                  Max. pool             f=256
[227,227,3] ----------> [55,55,96] ---------> [27,27,96] -------> [27,27,256] ...
             k=[11,11]             k=[3,3]               k=[5,5]
             s=4                   s=2                   s=1 
             p=0                                         p=2
             a='ReLU'                                    a='ReLU'
            Max. pool              f=384              f=384
[27,27,256] ---------> [13,13,256] -----> [13,13,384] ----------> [13,13,384] ...
            k=[3,3]                k=[3,3]            k=[3,3]
            s=2                    s=1                s=1 
                                   p=1                p=1
                                   a='ReLU'           a='ReLU'
            f=256                  Max. pool              FC
[13,13,384] ---------> [13,13,256] ----------> [6,6,256] ------> [4096,1] ...
            k=[3,3]                k=[3,3]
            s=1                    s=2
            p=1                    
            a='ReLU'               
           FC               FC
[4096,1]  ------> [4096,1] ----> softmax([1000,1])
'''
'''
print('Counting # of parameters in the Alex Net')
print('f=96,k=[11,11],s=4,p=0')
w1 = 3*11*11*96
print('w1=',w1)
print('f=256,k=[5,5],s=1,p=2')
w2 = (5*5*96)*256
print('w2=',w2)
print('f=384,k=[3,3],s=1,p=1')
w3 = (3*3*256)*384
print('w3=',w3)
print('f=384,k=[3,3],s=1,p=1')
w4 = (3*3*384)*384
print('w4=',w4)
print('f=256,k=[3,3],s=1,p=1')
w5 = (3*3*384)*256
print('w5=',w5)
print('FC [6,6,256]x4096')
w6 = 6*6*256*4096
print('w6=',w6)
print('FC 4096x4096')
w7 = 4096*4096
print('w7=',w7)
print('FC 4096x1000')
w8 = 4096*1000 + 1000
print('w8=',w8)
print('Total number of parameters:',w1+w2+w3+w4+w5+w6+w7+w8)
'''

