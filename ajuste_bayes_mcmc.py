import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("notas_andes.dat", skiprows = 1)

x_obs1 = data[:,0] 

x_obs2 = data[:,1]

x_obs3 = data[:,2]

x_obs4 = data[:,3]

y_obs = data[:,4]

sigma_y = 0.1

def modelo(x1,x2,x3,x4,b0,b1,b2,b3,b4):
    return x1*b1 + x2*b2 + x3*b3 + x4*b4 + b0

def log_verosimil(x1,x2,x3,x4,b0,b1,b2,b3,b4,y_obs,sigma_y):
    d = y_obs - modelo(x1,x2,x3,x4,b0,b1,b2,b3,b4)
    d = d/sigma_y
    d = -0.5*np.sum(d**2)
    return d

def log_prior(b0,b1,b2,b3,b4):
    p = -np.inf
    if b0 < 10 and b0 > 0 and b1 < 10 and b1 > 0  and b2 < 10 and b2 > 0 and b3 < 10 and b3 > 0 and b4 < 10 and b4 > 0:
        p = 0.0
    return p

N = 20000

lista_b0 = [np.random.random()]
lista_b1 = [np.random.random()]
lista_b2 = [np.random.random()]
lista_b3 = [np.random.random()]
lista_b4 = [np.random.random()]

log_post = [log_verosimil(x_obs1,x_obs2,x_obs3,x_obs4,lista_b0[0],lista_b1[0],lista_b2[0],lista_b3[0],lista_b4[0],y_obs,sigma_y)+log_prior(lista_b0[0],lista_b1[0],lista_b2[0],lista_b3[0],lista_b4[0])]

sigma_b0 = 0.005
sigma_b1 = 0.005
sigma_b2 = 0.005
sigma_b3 = 0.005
sigma_b4 = 0.005

for i in range(1,N):
    
    propuesta_b0 = lista_b0[i-1]+np.random.normal(loc = 0.0, scale = sigma_b0)
    propuesta_b1 = lista_b1[i-1]+np.random.normal(loc = 0.0, scale = sigma_b1)
    propuesta_b2 = lista_b2[i-1]+np.random.normal(loc = 0.0, scale = sigma_b2)
    propuesta_b3 = lista_b3[i-1]+np.random.normal(loc = 0.0, scale = sigma_b3)
    propuesta_b4 = lista_b4[i-1]+np.random.normal(loc = 0.0, scale = sigma_b4)
    
    log_post_anterior = log_verosimil(x_obs1,x_obs2,x_obs3,x_obs4,lista_b0[i-1],lista_b1[i-1],lista_b2[i-1],lista_b3[i-1],lista_b4[i-1],y_obs,sigma_y)+log_prior(lista_b0[i-1],lista_b1[i-1],lista_b2[i-1],lista_b3[i-1],lista_b4[i-1])
    log_post_actual = log_verosimil(x_obs1,x_obs2,x_obs3,x_obs4,propuesta_b0,propuesta_b1,propuesta_b2,propuesta_b3,propuesta_b4,y_obs,sigma_y)+log_prior(propuesta_b0,propuesta_b1,propuesta_b2,propuesta_b3,propuesta_b4)
    
    r = min(1,np.exp(log_post_actual-log_post_anterior))    
    alpha = np.random.random()
    if(alpha<r):
        lista_b0.append(propuesta_b0)
        lista_b1.append(propuesta_b1)
        lista_b2.append(propuesta_b2)
        lista_b3.append(propuesta_b3)
        lista_b4.append(propuesta_b4)
        log_post.append(log_post_actual)
    else:
        lista_b0.append(lista_b0[i-1])
        lista_b1.append(lista_b1[i-1])
        lista_b2.append(lista_b2[i-1])
        lista_b3.append(lista_b3[i-1])
        lista_b4.append(lista_b4[i-1])
        log_post.append(log_post_anterior)
        
lista_b0 = np.array(lista_b0)
lista_b1 = np.array(lista_b1)
lista_b2 = np.array(lista_b2)
lista_b3 = np.array(lista_b3)
lista_b4 = np.array(lista_b4)
log_post = np.array(log_post)

plt.figure(figsize = (10,5))

plt.subplot(2,3,1)
histo = plt.hist(lista_b0[10000:], bins=15, density=True)
plt.xlabel(r'$\beta_0$')
plt.title(r'$\beta$={:.2f}$\pm${:.2f}'.format(np.mean(lista_b0),np.std(lista_b0)))

plt.subplot(2,3,2)
histo = plt.hist(lista_b1[10000:], bins=15, density=True)
plt.xlabel(r'$\beta_1$')
plt.title(r'$\beta$={:.2f}$\pm${:.2f}'.format(np.mean(lista_b1),np.std(lista_b1)))

plt.subplot(2,3,3)
histo = plt.hist(lista_b2[10000:], bins=15, density=True)
plt.xlabel(r'$\beta_2$')
plt.title(r'$\beta$={:.2f}$\pm${:.2f}'.format(np.mean(lista_b2),np.std(lista_b2)))

plt.subplot(2,3,4)
histo = plt.hist(lista_b3[10000:], bins=15, density=True)
plt.xlabel(r'$\beta_3$')
plt.title(r'$\beta$={:.2f}$\pm${:.2f}'.format(np.mean(lista_b3),np.std(lista_b3)))

plt.subplot(2,3,5)
histo = plt.hist(lista_b4[10000:], bins=15, density=True)
plt.xlabel(r'$\beta_4$')
plt.title(r'$\beta$={:.2f}$\pm${:.2f}'.format(np.mean(lista_b4),np.std(lista_b4)))

plt.tight_layout()

plt.savefig('ajuste_bayes_mcmc.png')
