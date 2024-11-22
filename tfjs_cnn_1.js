import * as tf from '@tensorflow/tfjs';

// Load and preprocess the MNIST dataset
const mnist = tf.data.mnist();

const [trainXs, trainYs] = mnist.train.next();
const [testXs, testYs] = mnist.test.next();

// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.conv2d({
  filters: 32,
  kernelSize: 3,
  activation: 'relu',
  inputShape: [28, 28, 1]
}));
model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

// Compile the model
model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});

// Train the model
model.fit(trainXs, trainYs, { epochs: 5, batchSize: 32 }).then(() => {
  // Evaluate the model
  const testAcc = model.evaluate(testXs, testYs);
  console.log('Test accuracy:', testAcc);
});