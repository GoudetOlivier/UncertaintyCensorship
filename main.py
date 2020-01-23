from fn import *

##########
#  Main  #
##########

sample_size = 200
nb_iter = 100

error_threshold = 0.01

layers_size=[2,10,1]

a0 = 1
a1 = 0
a2 = 0

b0 = 1
b1 = 0
b2 = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"

surv, censor, obs, delta,  xi, x = gen_data(nb_iter,sample_size, a0, a1, a2, b0, b1, b2)

net = Net(layers_size)
net.to(torch.double)
net.to(device)

optimizer = torch.optim.SGD(net.parameters(),lr=.1)

criterion = nn.MSELoss()

#inputs = torch.tensor([obs,x],dtype=torch.double,device=device)
#target = torch.tensor(delta,dtype=torch.double,device=device)
outputs_batch = torch.ones(sample_size,dtype=torch.double,device=device)

for k in range(nb_iter):
	error = 1000
	loop_c = 0

	inputs_batch = torch.tensor([obs[k,], x[k,]],dtype=torch.double,device=device)
	target_batch = torch.tensor(delta[k,],dtype=torch.double,device=device)

	while error > error_threshold:
		
		loop_c += 1

		for i in range(sample_size):
			outputs_batch[i] = net.forward(inputs_batch[:,i])

		loss = criterion(outputs_batch,target_batch)
		error = loss.item()

		loss.backward(retain_graph=True)
			
		optimizer.step()
		optimizer.zero_grad()

		print(k,loss)
