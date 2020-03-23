from fn import *
from tqdm import tqdm
from os import system
import matplotlib.pyplot as plt
import math

##########
#  Main  #
##########
sample_size = 1000
nb_iter = 100
nb_epoch = 200

error_threshold = 0.01

layers_size = [2,1000,1]
x_list = [.3,.5,.7]

a0 = 3
a1 = .5
a2 = .7

b0 = 1
b1 = .5
b2 = .4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"

surv, censor, obs, delta,  xi, x = gen_data(nb_iter,sample_size, a0, a1, a2, b0, b1, b2)

p = np.zeros((nb_iter,sample_size))
###################
# Neural Network  #
###################
print('Neural network training : sample size =',sample_size,'iteration =',nb_iter,'epoch =',nb_epoch)
net = Net(layers_size)
# net.to(torch.float)
net.to(device)

optimizer = torch.optim.SGD(net.parameters(),lr=.001)
criterion = nn.L1Loss()

#inputs = torch.tensor([obs,x],dtype=torch.double,device=device)
#target = torch.tensor(delta,dtype=torch.double,device=device)
outputs_batch = torch.ones(sample_size,dtype=torch.float,device=device)

pbar = tqdm(range(nb_iter))

for k in pbar:

	inputs_batch = torch.transpose(torch.tensor([obs[k,], x[k,]],dtype=torch.float,device=device),0,1)
	target_batch = torch.tensor(delta[k,],dtype=torch.float,device=device)

	target_batch = target_batch.view(-1,1)

	net.reset_parameters()

	for epoch in range(nb_epoch):
		
		outputs_batch = net(inputs_batch)

		loss = criterion(outputs_batch,target_batch)

		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

	p[k,:] = outputs_batch.cpu().data.numpy()[:,0]

	side = ((target_batch-outputs_batch)**2).mean().item(),2*(target_batch*(target_batch-outputs_batch)).mean().item()

	pbar.set_postfix(loss=loss.item())

#######################
# Survival estimation #
#######################
print('Survival estimators computing')

t = np.linspace(np.amin(surv),np.amax(surv),num=100)
beran = np.zeros((nb_iter,len(t),len(x_list)))
nn_beran = np.zeros((nb_iter,len(t),len(x_list)))

for k in pbar:

	#Bandwidth selection
	h = .1
	c_x = 0
	for x_eval in x_list:
		#Estimators computation
		beran[k,:,c_x] = gene_Beran(t,surv[k,:],delta[k,:],x[k,:],x_eval,h)
		nn_beran[k,:,c_x] = gene_Beran(t,surv[k,:],p[k,:],x[k,:],x_eval,h)

		c_x += 1

np.save('save/beran',beran)
np.save('save/nn_beran',nn_beran)

################
# Plot results #
################
plt.figure()

for i in range(len(x_list)):

	true_cdf = expon(scale=1/(a0+a1*x_list[i]+a2*x_list[i]**2)).cdf(t)

	mean_beran = np.mean(beran[:,:,i],axis=0)
	mean_nn_beran = np.mean(nn_beran[:,:,i],axis=0)

	mise_beran = np.mean((beran[:,:,i]-true_cdf)**2,axis=0)
	mise_nn_beran = np.mean((nn_beran[:,:,i]-true_cdf)**2,axis=0)

	plt.subplot(len(x_list),2,2*i+1)
	plt.plot(t,mean_beran,label='Beran')
	plt.plot(t,mean_nn_beran,label='NN Beran')
	plt.plot(t,true_cdf,label='cdf')

	plt.legend()

	plt.subplot(len(x_list),2,2*(i+1))
	plt.plot(t,mise_beran,label='Beran')
	plt.plot(t,mise_nn_beran,label='NN Beran')

	plt.legend()
plt.show()