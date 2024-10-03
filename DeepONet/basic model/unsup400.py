import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:2')
print(device)
class MyClass():
    def __init__(self, param):
        self.param = param

def save_object(obj, directory, filename):
    try:
        # Create the full path by concatenating the directory and filename
        full_path = os.path.join(directory, filename)
        with open(full_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
class DeepOnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,input_size1, hidden_size1, output_size1):
        super(DeepOnet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        self.fc3 = nn.Linear(input_size1, hidden_size1)
        self.fc4 = nn.Linear(hidden_size1, output_size1)
    def forward(self, x,xx):
        

        
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        
        xx=torch.tanh(self.fc3(xx))
        xx = self.fc4(xx)
        x = torch.einsum("bi,ni->bn", x, xx)
        
        return x

p=40
input_size = 400
hidden_size = 40
output_size = p
input_size1 = 1
hidden_size1 = 40
output_size1 = p


model = DeepOnet(input_size, hidden_size, output_size,input_size1, hidden_size1, output_size1).to(device)
criterion = nn.MSELoss()  # You can choose an appropriate loss function
#optimizer = optim.SGD(model.parameters(), lr=0.01)  # You can choose the optimizer and learning rate
optimizer = optim.Adam(model.parameters(), lr=0.00001)

d = np.load("1000train.npz", allow_pickle=True)
train_x = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y = d["y"].astype(np.float32)
x = train_x[0]
xx = train_x[1]
x = torch.from_numpy(x)
xx = torch.from_numpy(xx)
x = x.to(device)
xx = xx.to(device)
#xx = xx.clone().detach().requires_grad_(True)
N=len(x[0])
h=1/(N-1)
A=np.zeros((N,N))
i,j = np.indices(A.shape)
A[i==j] = 2
A[i==j-1] = -1
A[i==j+1] = -1
A=A/(h**2)
A=torch.from_numpy(A)
A=A.to(device)
A=A.float()
y = torch.from_numpy(y)
y = y.to(device)
targets=y
#print(inputs)

print("N is", N)


d = np.load("1000test.npz", allow_pickle=True)
train_xtest = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
ytest = d["y"].astype(np.float32)
yytest = d["yy"].astype(np.float32)
xtest = train_xtest[0]
xxtest = train_xtest[1]
xtest = torch.from_numpy(xtest)
xxtest = torch.from_numpy(xxtest)
xtest = xtest.to(device)
xxtest = xxtest.to(device)
#xx = xx.clone().detach().requires_grad_(True)
ytest = torch.from_numpy(ytest)
ytest = ytest.to(device)
targetstest=ytest


print("number of data pairs y =",len(y))
#print("number of data pairs y1 =",len(y1))
print("number of data pairs ytest =",len(ytest))
#print("number of data pairs ytest1 =",len(ytest1))

num_epochs =50001
yes1=[]
yesn1=[]

yes=[]
yesn=[]

yestest=[]
yesntest=[]

yestest1=[]
yesntest1=[]


yest=[]


when=[]
lamda=0.5
tell=0

total_time_task1=0
time1=[]
directory="/home/skl5876/Data_set/NN/"
filename = 'data.unsup'
for epoch in range(num_epochs):
    start_time_task1 = time.time()

    
    outputs = model(x,xx)  # inputs are your input data
    #print(outputs)
    x123=torch.transpose(outputs , 0, 1).float()
    aa1=(A@x123)
    aa1=torch.transpose(aa1, 0, 1)
    aa=torch.mul(outputs,2*x)+aa1
    x223=torch.transpose(x , 0, 1).float()
    aa2=(A@x223)
    aa2=torch.transpose(aa2, 0, 1)
    bb=-aa2-torch.mul(x,x)
    

    yes.append(criterion(outputs, targets).item())
    yesn.append(criterion(aa[:,1:-1] ,bb[:,1:-1]).item())

    
    

    loss =( criterion(aa[:,1:-1] ,bb[:,1:-1])+1000*(criterion( outputs[:,0],-x[:,0])+criterion( outputs[:,-1],-x[:,-1]+1)))  # targets are your target values

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    yest.append(loss.item())
    
    outputstest = model(xtest,xxtest)  # inputs are your input data
    #print(outputs)
    x123=torch.transpose(outputstest , 0, 1).float()
    aa1=(A@x123)
    aa1=torch.transpose(aa1, 0, 1)
    aa=torch.mul(outputstest,2*xtest)+aa1
    x223=torch.transpose(xtest , 0, 1).float()
    aa2=(A@x223)
    aa2=torch.transpose(aa2, 0, 1)
    bb=-aa2-torch.mul(xtest,xtest)
    

    yestest.append(criterion(outputstest, targetstest).item())
    yesntest.append(criterion(aa[:,1:-1] ,bb[:,1:-1]).item())


    if epoch % 10000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

        print("total time is ",total_time_task1)

        plt.plot(yes, label="'MSE_loss'")
        plt.plot(yesn, label="'Newton_loss'")
        #plt.plot(yes1, label="'MSE_loss1'")
        #plt.plot(yesn1, label="'Newton_loss1'")
        plt.plot(yestest, label="'MSE_losstest'")
        plt.plot(yesntest, label="'Newton_losstest'")
        plt.plot(yestest1, label="'MSE_losstest1'")
        plt.plot(yesntest1, label="'Newton_losstest1'")
        plt.plot(yest, label="'Train_loss'")
        plt.vlines(x = when, ymin = min(min(yes),min(yesn)) , ymax = max(max(yes),max(yesn)), color = 'b')
        plt.legend(loc="upper left")
        plt.yscale("log")
        plt.legend()
        plt.legend(loc='upper left')
        plt.show()










        torch.save(model, os.path.join(directory,  'deep_onet_modelunsup.pth'))

        # Save the model's state dictionary (recommended)
        torch.save(model.state_dict(), os.path.join(directory,  'deep_onet_model_state_dictunsup.pth'))

        # Save the optimizer's state (optional)
        torch.save(optimizer.state_dict(), os.path.join(directory, 'deep_onet_optimizer_state_dictunsup.pth'))

            


        obj = MyClass((yes,yesn,yes1,yesn1,yestest,yesntest,yestest1,yesntest1,yest,when,lamda,time1))
        # Save the object
        save_object(obj, directory, filename)


    elapsed_time_task1 = time.time() - start_time_task1
    total_time_task1 += elapsed_time_task1
    time1.append(total_time_task1)
            
            
torch.save(model, os.path.join(directory,  'deep_onet_modelunsup.pth'))

# Save the model's state dictionary (recommended)
torch.save(model.state_dict(), os.path.join(directory,  'deep_onet_model_state_dictunsup.pth'))

# Save the optimizer's state (optional)
torch.save(optimizer.state_dict(), os.path.join(directory, 'deep_onet_optimizer_state_dictunsup.pth'))





obj = MyClass((yes,yesn,yes1,yesn1,yestest,yesntest,yestest1,yesntest1,yest,when,lamda,time1))
# Save the object
save_object(obj, directory, filename)





























num_epochs =300001
optimizer = optim.Adam(model.parameters(), lr=0.000001)
for epoch in range(num_epochs):
    start_time_task1 = time.time()

    
    outputs = model(x,xx)  # inputs are your input data
    #print(outputs)
    x123=torch.transpose(outputs , 0, 1).float()
    aa1=(A@x123)
    aa1=torch.transpose(aa1, 0, 1)
    aa=torch.mul(outputs,2*x)+aa1
    x223=torch.transpose(x , 0, 1).float()
    aa2=(A@x223)
    aa2=torch.transpose(aa2, 0, 1)
    bb=-aa2-torch.mul(x,x)
    

    yes.append(criterion(outputs, targets).item())
    yesn.append(criterion(aa[:,1:-1] ,bb[:,1:-1]).item())

    

    
    loss =( criterion(aa[:,1:-1] ,bb[:,1:-1])+1000*(criterion( outputs[:,0],-x[:,0])+criterion( outputs[:,-1],-x[:,-1]+1)))  # targets are your target values

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    yest.append(loss.item())
    
    outputstest = model(xtest,xxtest)  # inputs are your input data
    #print(outputs)
    x123=torch.transpose(outputstest , 0, 1).float()
    aa1=(A@x123)
    aa1=torch.transpose(aa1, 0, 1)
    aa=torch.mul(outputstest,2*xtest)+aa1
    x223=torch.transpose(xtest , 0, 1).float()
    aa2=(A@x223)
    aa2=torch.transpose(aa2, 0, 1)
    bb=-aa2-torch.mul(xtest,xtest)
    

    yestest.append(criterion(outputstest, targetstest).item())
    yesntest.append(criterion(aa[:,1:-1] ,bb[:,1:-1]).item())




    if epoch % 10000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

        print("total time is ",total_time_task1)

        plt.plot(yes, label="'MSE_loss'")
        plt.plot(yesn, label="'Newton_loss'")
        #plt.plot(yes1, label="'MSE_loss1'")
        #plt.plot(yesn1, label="'Newton_loss1'")
        plt.plot(yestest, label="'MSE_losstest'")
        plt.plot(yesntest, label="'Newton_losstest'")
        plt.plot(yestest1, label="'MSE_losstest1'")
        plt.plot(yesntest1, label="'Newton_losstest1'")
        plt.plot(yest, label="'Train_loss'")
        plt.vlines(x = when, ymin = min(min(yes),min(yesn)) , ymax = max(max(yes),max(yesn)), color = 'b')
        plt.legend(loc="upper left")
        plt.yscale("log")
        plt.legend()
        plt.legend(loc='upper left')
        plt.show()








        torch.save(model, os.path.join(directory,  'deep_onet_modelunsup.pth'))

        # Save the model's state dictionary (recommended)
        torch.save(model.state_dict(), os.path.join(directory,  'deep_onet_model_state_dictunsup.pth'))

        # Save the optimizer's state (optional)
        torch.save(optimizer.state_dict(), os.path.join(directory, 'deep_onet_optimizer_state_dictunsup.pth'))






        obj = MyClass((yes,yesn,yes1,yesn1,yestest,yesntest,yestest1,yesntest1,yest,when,lamda,time1))
        # Save the object
        save_object(obj, directory, filename)


    elapsed_time_task1 = time.time() - start_time_task1
    total_time_task1 += elapsed_time_task1
    time1.append(total_time_task1)
            
            
torch.save(model, os.path.join(directory,  'deep_onet_modelunsup.pth'))

# Save the model's state dictionary (recommended)
torch.save(model.state_dict(), os.path.join(directory,  'deep_onet_model_state_dictunsup.pth'))

# Save the optimizer's state (optional)
torch.save(optimizer.state_dict(), os.path.join(directory, 'deep_onet_optimizer_state_dictunsup.pth'))





obj = MyClass((yes,yesn,yes1,yesn1,yestest,yesntest,yestest1,yesntest1,yest,when,lamda,time1))
# Save the object
save_object(obj, directory, filename)






