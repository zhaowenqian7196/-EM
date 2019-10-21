from __future__ import print_function
import numpy as np


def generateData(k,mu,sigma,dataNum):
    dataArray = np.zeros(dataNum,dtype=np.float32)
    n = len(k)
    for i in range(dataNum):
        rand = np.random.random()
        Sum = 0
        index = 0
        while(index < n):
            Sum += k[index]
            if(rand < Sum):
                dataArray[i] = np.random.normal(mu[index],sigma[index])
                break
            else:
                index += 1
    return dataArray

def normPdf(x,mu,sigma):
    return (1./np.sqrt(2*np.pi))*(np.exp(-(x-mu)**2/(2*sigma**2)))



def em(dataArray,k,mu,sigma,step = 50):
    n = len(k)
    dataNum = dataArray.size
    gamaArray = np.zeros((n,dataNum))
    for s in range(step):
        for i in range(n):
            for j in range(dataNum):
                Sum = sum([k[t]*normPdf(dataArray[j],mu[t],sigma[t]) for t in range(n)])
                gamaArray[i][j] = k[i]*normPdf(dataArray[j],mu[i],sigma[i])/float(Sum)

        for i in range(n):
            mu[i] = np.sum(gamaArray[i]*dataArray)/np.sum(gamaArray[i])
  
        for i in range(n):
            sigma[i] = np.sqrt(np.sum(gamaArray[i]*(dataArray - mu[i])**2)/np.sum(gamaArray[i]))
  
        for i in range(n):
            k[i] = np.sum(gamaArray[i])/dataNum

    return [k,mu,sigma]





if __name__ == '__main__':

    k = [0.3,0.4,0.3]
    mu = [2,4,3]
    sigma = [1,1,4]

    dataNum = 5000

    dataArray = generateData(k,mu,sigma,dataNum)

    k0 = [0.3,0.3,0.4]
    mu0 = [1,2,2]
    sigma0 = [1,1,1]
    step = 10

    k1,mu1,sigma1 = em(dataArray,k0,mu0,sigma0,step)

    print("参数实际值:")
    print("k:",k)
    print("mu:",mu)
    print("sigma:",sigma)
    print("参数估计值:")
    print("k1:",k1)
    print("mu1:",mu1)
    print("sigma1:",sigma1)