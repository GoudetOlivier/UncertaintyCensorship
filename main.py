##########
#  Main  #
##########

n_loop = 50
size = 100

#np.random.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
x = np.random.uniform(low=0.0, high=1.0, size=size)
y = 2*x+1

inputs = torch.tensor(x,dtype=torch.double,device=device)
output = torch.ones(size,dtype=torch.double,device=device)
target = torch.tensor(y,dtype=torch.double,device=device)

net = Net(1,1,3,10)
net.to(torch.double)
net.to(device)

optimizer = torch.optim.SGD(net.parameters(),lr=.1)

criterion = nn.MSELoss()

for k in range(n_loop):

    optimizer.zero_grad()

    for i in range(size):
        output[i] = net.forward(inputs[i])

    loss = criterion(output,target)
    print(k,loss.item())
    loss.backward(retain_graph=True)
    optimizer.step()
