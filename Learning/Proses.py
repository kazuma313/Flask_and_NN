from Learning import nn, Layer

def learn(x_train, y_train, hidden1:int=6, hidden2:int=5, learningRate:float=0.02, epoch:int=500):
    nn.add_layer(Layer(14, hidden1, 'sigmoid'))
    nn.add_layer(Layer(hidden1, hidden2, 'sigmoid'))
    nn.add_layer(Layer(hidden2, 1, 'sigmoid'))
    errors = nn.train(x_train.T, y_train, learningRate, epoch)
    
    return errors