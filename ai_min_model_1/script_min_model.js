/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const summary = document.getElementById('summary');
const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

// cocoSsd.load().then(function (loadedModel) {
//     model = loadedModel;
//     // Show demo section now model is ready to use.
//     demosSection.classList.remove('invisible');
//   });

//   const MODEL_PATH_LOC = './model.json';

/********LOAD MOVENET ************* */
const EXAMPLE_IMG = document.getElementById('exampleImg')
const MODEL_PATH_move = 'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4';
let movenet = undefined;
async function loadAndRunModel(){
    movenet = await tf.loadGraphModel(MODEL_PATH_move, {fromTFHub:true});
    let exampleInputTensor = tf.zeros([1,192,192,3],'int32'); //MOCK black img
    let imageTensor=tf.browser.fromPixels(EXAMPLE_IMG);
    console.log(imageTensor.shape);
    //BEWARE Shape mismatch y,x.
    let cropStartPoint=[15,170,0]; //y,x,color
    let cropSize=[345,345,3]///tgt img size
    //slices tensor, to the cropped size.
    let croppedTensor=tf.slice(imageTensor,cropStartPoint,cropSize);
    let resizedTensor=tf.image.resizeBilinear(croppedTensor,[192,192],true).toInt();
    console.log(resizedTensor.shape);
    
    //expandDims wraps with a 1 [] dimension.
    let tensorOutput = movenet.predict(exampleInputTensor);
    let arrayOutput = await tensorOutput.array();
    console.log(arrayOutput);
    //DISPOSE TENSORS!!!
    tensorOutput.dispose();
    resizedTensor.dispose();
    croppedTensor.dispose();
    imageTensor.dispose();
    imageTensor.dispose();
}
// loadAndRunModel();

/********LOAD BY URL ************* */
const MODEL_PATH_URL = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json';
let model = undefined;
async function loadModel(){
    model = await tf.loadLayersModel(MODEL_PATH_URL);
    model.summary();
    // create a batch of 1
    const input = tf.tensor2d([[870]]);
    // create a batch of 3
    const inputBatch = tf.tensor2d([[500], [1100], [970]]);
    //make prediction for each batch, can optimize here parallel
    const result = model.predict(input);
    const resultBatch = model.predict(inputBatch);
    //NOTE use to Array to get JS version of these MODELS.
    const arr1 = await result.array();
    const arr2 = await resultBatch.array();

    //SAVE BATCH MODEL to LOCALSTORAGE
    // await model.save('localstorage://demo/aiModelTest1')
    //LOAD MODEL from LOCALSTORAGE
    // console.log(JSON.stringify(await tf.io.listModels()));
    // let localM = await tf.loadLayersModel('localstorage://demo/aiModelTest1');
    // localM.dispose();
    result.print();
    resultBatch.print();
    input.dispose();
    inputBatch.dispose();
    result.dispose();
    resultBatch.dispose();
    model.dispose();
}
//   loadModel();

/********LOAD BY URL ************* */
const MODEL_PATH_LS = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json';
let model_min = undefined;
async function loadModel(){
    model_min = await tf.loadLayersModel(MODEL_PATH_LS);
    model_min.summary();
    if (summary && model_min) {
        summary.innerText = 'Summary: ' ;
    }    
}
//   loadModel();

/********LOAD BY BROWSER FILE ************* */

  // Note: this code snippet will not work without the HTML elements in the
//   page
// const jsonUpload = document.getElementById('json-upload');
// const weightsUpload = document.getElementById('weights-upload');

// const model = await tf.loadLayersModel(
//      tf.io.browserFiles([jsonUpload.files[0], weightsUpload.files[0]]));

/********LOAD BY LOCAL STORAGE ************* */
//      const model = tf.sequential(
//         {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
//    console.log('Prediction from original model:');
//    model.predict(tf.ones([1, 3])).print();
   
//    const saveResults = await model.save('localstorage://my-model-1');
   
//    const loadedModel = await tf.loadLayersModel('localstorage://my-model-1');
//    console.log('Prediction from loaded model:');
//    loadedModel.predict(tf.ones([1, 3])).print();

