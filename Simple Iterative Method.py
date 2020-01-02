# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 18:02:46 2019

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import xlsxwriter as xlw
import openpyxl as oxl
import scipy.optimize as spo
import scipy.integrate as sci
import quadpy


from scipy.stats import norm
from mpl_toolkits .mplot3d import Axes3D 
from matplotlib import cm 
from matplotlib .ticker import LinearLocator , FormatStrFormatter
from pandas_datareader import data
from openpyxl import load_workbook

def CRRPut(S0,K,r,sig,T,N):
    """
    This is to calculate American Put option using Binomial tree model
        
    """
    ST_1=[0]*(N+1)
    ST_0=[0]*(N+1)
    PnL_1=[0]*(N+1)
    PnL_0=[0]*(N+1)
    dt=T/N
    u=np.exp(sig*np.sqrt(dt))
    d=1/u
    pu=(np.exp(r*dt)-d)/(u-d)
    pd=1-pu
    
    for i in range(N+1): # to generate stock prices at time T
        ST_1[i]=S0*(d**i)*(u**(N-i))
    for i in range(N+1):
        PnL_1[i]=max(K-ST_1[i],0) #intrinsic value of American put at time T
        
    for i in range(N):
        for j in range(N-i):
            PnL_0[j]=max(np.exp(-r*dt)*(pu*PnL_1[j]+pd*PnL_1[j+1]),max(K-ST_1[j]*d,0))
            ST_0[j]=ST_1[j]*d
        ST_1=ST_0
        PnL_1=PnL_0
        
    return PnL_0[0]

def NCDF(x):
    """
    Return the normal distribution cumulative function of value x
    """
    return norm.cdf(x)

def d12(S,T,B):
    """
    To calculate d1 and d2 as in Black-Scholes-Merton
    """
    d1=(np.log(S/B) + (r+0.5*sig**2)*T)/(sig*np.sqrt(T))
    d2=d1-sig*np.sqrt(T)
    return d1, d2

def BSPut(S0,K,r,sig,T):
    """
    This functionis to calculate European Put option price using BS formula
    """
    d1=(1/sig/np.sqrt(T))*(np.log(S0/K)+(r+0.5*sig**2)*T)
    d2=d1-sig*np.sqrt(T)
    N1=norm.cdf(-d1)
    N2=norm.cdf(-d2)
    P=N2*K*np.exp(-r*T)-N1*S0
    return P
    
def Polyf(params,x):
    """
    Polynomial order n of the optimal exercise boundaries
    """
    sum0=0
    for l in range(len(params)):
        p=len(params)-1-l
        sum0+=params[l]*x**p
    return sum0

def MainMethod(S0,K,r,sig,T,k,n):
    """
    Main function to calculate American option price as detailed in the research paper
    """
    dt=T/n
    B=[K]*(n+1)
    #first iteration - equation (6) in paper
    for i in range(n):
        i+=1
        tt = i*dt
        A1 = K*(NCDF(d12(K,tt,K)[0])+(1/sig/np.sqrt(2*np.pi*tt))*np.exp(-0.5*d12(K,tt,K)[0]**2))**(-1)
        A2 = 1/sig/np.sqrt(2*np.pi*tt)*np.exp(-(r*tt+0.5*d12(K,tt,K)[1]**2))
        A3 = (2*sig*r)/(2*r+sig**2)*(2*NCDF((2*r+sig**2)*np.sqrt(tt)/(2*sig))-1)
        B[i]=A1*(A2+A3)
     
    for i in range(k): #kth iteration as in equation (5)
        xvals=np.linspace(0,T,n+1)
        params=np.polyfit(xvals,B,deg=n)
        
        for j in range(n): #equation (5) at each node point
            j+=1
            tt = j*dt
            A1 = (NCDF(d12(B[j],tt,K)[0])+(1/sig/np.sqrt(2*np.pi*tt))*np.exp(-0.5*d12(B[j],tt,K)[0]**2))**(-1)
            A2 = (1/sig/np.sqrt(2*np.pi*tt)*K*np.exp(-(r*tt+0.5*d12(B[j],tt,K)[1]**2)))
            A3 = r*K*sci.fixed_quad(lambda x: (1/sig/np.sqrt(2*np.pi*(tt-x)))*r*np.exp(-(r*(tt-x)+\
            0.5*d12(B[j],tt,Polyf(params,x))[1]**2)),0,tt)[0]
            B[j]=A1*(A2+A3)
    
    xvals=np.linspace(0,T,n+1)
    params=np.polyfit(xvals,B,deg=n)
    
    Price = BSPut(S0,K,r,sig,T) + sci.fixed_quad(lambda x: r*K*np.exp(-r*(T-x))*NCDF(-d12(S0,T-x,Polyf(params,x))[1]),0,T)[0]
    return Price

def Phi(S,T,Gamma,H,X):
    """
    Phi function in BS1993 formula
    """
    
    Lambda = -r + b*Gamma + 0.5*Gamma*(Gamma-1)*sig**2
    Kappa = 2*b/sig**2 + (2*Gamma - 1)
    d = - (np.log(S/H)+(b+(Gamma-0.5)*sig**2)*T)/(sig*np.sqrt(T))
    """
    V1 = np.exp(Lambda*T)*S**Gamma
    V2 = NCDF(-(np.log(S/H)+(b+(Gamma-0.5)*sig**2)*T)/sig/np.sqrt(T))
    V3 = -(X/S)**Kappa*NCDF(-(np.log((X**2)/(S*H))+(b+(Gamma-0.5)*sig**2)*T)/sig/np.sqrt(T))
    """
    V = np.exp(Lambda*T)*S**Gamma*(NCDF(d) - (X/S)**Kappa*NCDF(d-2*np.log(X/S)/sig/np.sqrt(T)))
    return V
    
    
def BS1993Call(S,K,r,sig,T,b):
    """
    Close-form solution of American Option using Bjerksund & Stensland 1993 Formula
    """
    
    Beta = (0.5 - b/sig**2)+ np.sqrt((b/sig**2-0.5)**2+2*r/sig**2)
    B_inf = Beta/(Beta-1)*K
    B_0 = max(K,r/(r-b)*K)
    h = - (b*T + 2*sig*np.sqrt(T))*(B_0/(B_inf-B_0))
    X = B_0+(B_inf-B_0)*(1-np.exp(h))
    Alpha = (X-K)*X**(-Beta)
    
    #Main formula for American Call
    C = Alpha*S**Beta - Alpha*Phi(S,T,Beta,X,X) + Phi(S,T,1,X,X) - Phi (S,T,1,K,X) -\
    K*Phi(S,T,0,X,X) + K*Phi(S,T,0,K,X)
    return C
    
def QuadAppPut(S,K,r,sig,T,b):
    """
    Closed-form solution of American Option using Quadratic Approximation as in BW 1987
    """
    M = 2*r/sig**2
    N = 2*b/sig**2
    q1 = (-(N-1) - np.sqrt((N-1)**2 + 4*M/K))/2
    q2 = (-(N-1) + np.sqrt((N-1)**2 + 4*M/K))/2
    
    #S_Crit Calculation
    Err = 1
    count = 0
    S_Crit = K
    
    while (Err > 0.001) or (count<100):
        d1 = (np.log(S_Crit/K) + (b + 0.5*sig**2)*T)/sig/np.sqrt(T)
        a = (b-r)*T
        bi = np.exp(a)*NCDF(-d1)*(-1 + 1/q1) + (-1 - np.exp(a)*norm.pdf(NCDF(-d1))/sig/np.sqrt(T))/q1
        LHS = K - S_Crit
        RHS = BSPut(S_Crit,K,r,sig,T) - (1 - np.exp(a)*NCDF(-d1))*S_Crit/q1
        S_Crit = (K - RHS + bi*S_Crit)/(bi+1)
        Err = LHS - RHS
        count+=1
    
    """ TESTING Algo to find critical S
    while (Err > 0.01) or (count<100):
        d1 = (np.log(S_Crit/K) + (b + 0.5*sig**2)*T)/sig/np.sqrt(T)
        a = (b-r)*T
        RHS = BSPut(S_Crit,K,r,sig,T) - (1 - np.exp(a)*NCDF(-d1))*S_Crit/q1
        Err = K - S_Crit - RHS
        S_Crit = K - RHS
        count+=1
    """   
        
        
    d1 = (np.log(S_Crit/K) + (b + 0.5*sig**2)*T)/sig/np.sqrt(T)    
    A1 = -(S_Crit/q1)*(1 - np.exp(a)*NCDF(-d1))
    
    if (S > S_Crit):
        Price = BSPut(S,K,r,sig,T) + A1*(S/S_Crit)**q1
    else:
        Price = K - S
        
    return Price


"""                           Main Program                      """
S=45
sig=0.2
r=0.05
T=0.5
N=100
K=45

#Binomial Method
print(CRRPut(S,K,r,sig,T,N))

#Main Iterative Method
k=100
n=16
print(MainMethod(S,K,r,sig,T,k,n))

#Bjerksund & Stenland 1993 Method
b=r
print (BS1993Call(K,S,r-b,sig,T,-b))

#Quadratic Approximation 1987
print (QuadAppPut(S,K,r,sig,T,b))

            

