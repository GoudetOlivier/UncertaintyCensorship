from fn import *

##########
#  Main  #
##########

sample_size = 200
nb_iter = 100

error_threshold = 0.01

depth = 3
width = 10

a0 = 1
a1 = 10
a2 = 10

b0 = 0.5
b1 = 5
b2 = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

surv, censor, obs, delta,  xi, x = gen_data(nb_iter,sample_size, a0, a1, a2, b0, b1, b2)

net = Net(1,1,depth,width)
net.to(torch.double)
net.to(device)

optimizer = torch.optim.SGD(net.parameters(),lr=.1)

criterion = nn.MSELoss()

inputs = torch.tensor(x,dtype=torch.double,device=device)
output = torch.ones([nb_iter,sample_size],dtype=torch.double,device=device)
target = torch.tensor(censor,dtype=torch.double,device=device)

for k in range(nb_iter):
	error = 1000
	loop_c = 0
	while error > error_threshold:
		
		loop_c += 1

		for i in range(sample_size):
			output[k,i] = net.forward(inputs[k,i])

		loss = criterion(output,target)
		error = loss.item()
		loss.backward(retain_graph=True)
		
		if np.mod(loop_c,8)==0:
			optimizer.step()
			optimizer.zero_grad()


		print(k,error)
