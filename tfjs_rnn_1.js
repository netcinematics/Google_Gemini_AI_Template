import * as tf from '@tensorflow/tfjs';

// Generate some sample time series data
const timeSeriesData = [];
for (let i = 0; i < 100; i++) {
  timeSeriesData.push(Math.sin(i / 10) + Math.random() * 0.2);
}

// Prepare the data for training
const sequences = [];
const labels = [];
for (let i = 0; i < timeSeriesData.length - 10; i++) {
  sequences.push(timeSeriesData.slice(i, i + 10));
  labels.push(timeSeriesData[i + 10]);
}

const xs = tf.tensor(sequences);
const ys = tf.tensor(labels);

// Define the RNN model
const model = tf.sequential();
model.add(tf.layers.lstm({ units: 32, inputShape: [10, 1] }));
model.add(tf.layers.dense({ units: 1 }));

// Compile the model
model.compile({
  optimizer: 'adam',
  loss: 'meanSquaredError'
});

// Train the model
model.fit(xs, ys, { epochs: 10 }).then(() => {
  // Make predictions
  const futureInput = tf.tensor([timeSeriesData.slice(-10)]);
  const futurePrediction = model.predict(futureInput);
  console.log('Predicted next value:', futurePrediction.dataSync()[0]);
});