from gurobipy import *

# Create a new model
model = Model("optimize tasks/s")

#----------------------------------------------------------
# Initialize LP variables
#----------------------------------------------------------

nbGpus  = 2
nbTasks = 2

#Bandwidth
B = 350

#Size of the task i
W_i = [100, 250]

#Memory of the GPU j
M_j = [300, 300]

#Task i computing time for GPU j
Y_ij = [[0.6, 0.7],[0.8, 0.9]]

#Entry size of W_i
A_i = [100, 250]


#----------------------------------------------------------
# Initialize LP variables
#----------------------------------------------------------

#If the GPU j store W_i
U_ij = {}
for task_id in range(nbTasks):
    for gpu_id in range(nbGpus):
        U_ij[task_id, gpu_id] = model.addVar(vtype = GRB.BINARY)


#Number of tasks/s type of i found on the GPU j
X_ij = {}
for task_id in range(nbTasks):
    for gpu_id in range(nbGpus):
        X_ij[task_id, gpu_id] = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)

#A task
N_i = {}
for task_id in range(nbTasks):
    N_i[task_id] = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)

#----------------------------------------------------------
# Add constraints
#----------------------------------------------------------

#1)
minY_ij = min(min(Y_ij))
for task_id in range(nbTasks):
    for gpu_id in range(nbGpus):
        model.addConstr( X_ij[task_id, gpu_id] <= (U_ij[task_id, gpu_id] * 1/minY_ij))

#2)
for gpu_id in range(nbGpus):
    gpuMemorySize = LinExpr()
    for task_id in range(nbTasks):
        gpuMemorySize += W_i[task_id] * U_ij[task_id, gpu_id]
    model.addConstr(gpuMemorySize <= M_j[gpu_id])

#3)
for gpu_id in range(nbGpus):
    throughtPut = LinExpr()
    for task_id in range(nbTasks):
        throughtPut += Y_ij[task_id][gpu_id] * X_ij[task_id, gpu_id]
    model.addConstr(throughtPut <= 1.0)

#4)
for task_id in range(nbTasks):
    N = LinExpr()
    for gpu_id in range(nbGpus):
        N += X_ij[task_id, gpu_id]
    model.addConstr(N_i[task_id] == N)

#5)
NB = LinExpr()
for task_id in range(nbTasks):
    NB += N_i[task_id] + A_i[task_id]
model.addConstr(NB <= B)


#model.setObjective(X_ij, GRB.MAXIMIZE) #GRB.MAXIMIZE
model.optimize()

#Print values for decision variables
for v in model.getVars():
    print(v.varName, v.x)

#Print maximized profit value
print('Maximized profit:',  model.objVal)