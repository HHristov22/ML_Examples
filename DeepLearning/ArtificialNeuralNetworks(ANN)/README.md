# Training the ANN with Stochastic Gradient Descent

### Step 1:
Randomly initialize the weights to small numbers close to 0 (but not 0).

### Step 2:
Input the first observation of your dataset in the input layer, with each feature in one input node.

### Step 3:
**Forward-Propagation**: From left to right, the neurons are activated in a way that the impact of each neuron's activation is limited by the weights. Propagate the activations until the predicted result \( y \) is obtained.

### Step 4:
Compare the predicted result to the actual result. Measure the generated error.

### Step 5:
**Back-Propagation**: From right to left, the error is back-propagated. Update the weights according to how much they are responsible for the error. The learning rate decides by how much to update the weights.

### Step 6:
Repeat Steps 1 to 5 and update the weights after each observation (Stochastic Gradient Descent). Or, repeat Steps 1 to 5 but update the weights only after a batch of observations (Batch Learning).

### Step 7:
When the whole training set passes through the ANN, that completes an epoch. Redo more epochs.
