from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

# how wide our binary input will be, here we're only training and 
# testing with numbers up to 1024 so we only need 10 binary digits
bin_width = 10

# turns an array of numbers into fizzbuzz solutions, realistically 
# this is all we need to do, but whatever
def get_outputs(inputs):
	outputs = [None] * len(inputs)

	for i, x in enumerate(inputs):
		if x%3==0 and x%5==0:
			outputs[i] = "FizzBuzz"
		elif x%3==0:
			outputs[i] = "Fizz"
		elif x%5==0:
			outputs[i] = "Buzz"
		else:
			outputs[i] = x

	return outputs

# turns array of solutions into a 2d one-hot representation of the solutions
def encode(outputs):
	out = np.zeros([len(outputs),4])

	for ind, x in enumerate(outputs):
		if x == "FizzBuzz":
			out[ind][0] = 1
		elif x == "Fizz":
			out[ind][1] = 1
		elif x == "Buzz":
			out[ind][2] = 1
		else:
			out[ind][3] = 1

	return out

# truns array of numbers into a 2d array of binary digits of given numbers
def encode_bin(inputs):
	inp = np.zeros([len(inputs),bin_width])

	for ind, val in enumerate(inputs):
		inp[ind][bin_width-len(bin(val))+2:] = (np.array([int(x) for x in bin(val)[2:]]))

	return inp

# truns one-hot representation into fizzbuzz solutions
def decode(outputs):
	out = [None] * len(outputs)

	outputs = np.argmax(outputs, axis = 1, out = None)

	for ind, x in enumerate(outputs):
		if outputs[ind] == 0:
			out[ind] = "FizzBuzz"
		elif outputs[ind] == 1:
			out[ind] = "Fizz"
		elif outputs[ind] == 2:
			out[ind] = "Fizz"
		else:
			out[ind] = ind+1

	return np.array(out)

# Generates our deep learning model
def make_model():
	model = Sequential()
	model.add(Dense(32, input_shape=(bin_width,), activation = 'relu'))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(128, activation = 'relu'))
	model.add(Dense(128, activation = 'relu'))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(4, activation = 'softmax'))

	return model

# Creating testing data 
inputsN = encode_bin(np.arange(1, 101))		# testing with digits from 1-100

outputsN = encode(get_outputs(np.arange(1, 101)))

#Creating training data
input_train = encode_bin(np.arange(101,999)) # training with digits from 101-999

output_train = encode(get_outputs(np.arange(101,999)))

# Compiling and training our model
model = make_model()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(input_train, output_train, batch_size = 4, epochs=200, verbose=2, shuffle=True)

# Finally testing out model, at best I recived a 99% accuracy 
print("Score: %.2f%%" % (model.evaluate(inputsN, outputsN, verbose=0)[1]*100))

print(decode(model.predict(encode_bin(np.arange(1, 101)))))
