from discrete_neural_net import *

# myNN = RandomNeuralNet(5,('x','y','z'),4,3,40,{2: 7, 3: 4, 10: 3})
# print(myNN.feed_forward({'x': 0, 'y': 1, 'z':4}))

# architecture = [('x','y')]
# f = ModularAddition(10,cache_values=True)
# architecture.append(Layer([Node(f,['x','y']),Node(f,['x','y'])]))
# architecture.append(Layer([Node(f,[architecture[1].nodes[0],architecture[1].nodes[1]])]))
# myNN = NeuralNet(architecture,{1: [Identity(10)], 2: [f]})

def difference_loss(x,y):
    return abs(x[0]-y)

modulus = 10

myNN = RandomNeuralNet(modulus,('x','y'),1,6,4,{2: 2})
training_z = []
for u in range(modulus//2):
    for v in range(modulus//2):
        training_z.append(({'x': u, 'y': v},(u+v)%modulus))
# print(training_z)
for _ in range(1000):
#     myNN.random_train(training_z,difference_loss)
#    myNN.operation_tweak_train(training_z,difference_loss)
    myNN.activation_tweak_train(training_z,difference_loss)

successes = 0
for u in range(modulus):
    for v in range(modulus):
        if (u+v)%modulus == int((myNN.feed_forward({'x': u,'y': v}))[0]):
            successes += 1
        print(u,v,myNN.feed_forward({'x': u,'y': v}),(u+v)%modulus)
print(successes/(modulus**2))