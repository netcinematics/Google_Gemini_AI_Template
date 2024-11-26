// import {TRAINING_DATA} from 
// './real-estate-data.js';
// 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js'
//Input feature pairs (size, num, view area, zip)
// const INPUTS = TRAINING_DATA.inputs; //FEATURES
const INPUTS = []; //generate inputs
for(let n=1;n<=20;n++){ //EMULATE SMALL DATA SET (20)!!!
    INPUTS.push(n); //20*20=400 || largest output to predict???
}
const OUTPUTS = []; //squaring number;
for(let n=0;n<INPUTS.length;n++){
    OUTPUTS.push(INPUTS[n]*INPUTS[n]); //square
}
// use attributes to predict price
// const OUTPUTS = TRAINING_DATA.outputs; //PREDICTION VALUES
//shuffle - avoids INPUT PATTERNS. of Data Sequence.
// tf.util.shuffleCombo(INPUTS, OUTPUTS);
// const INPUTS_TENSOR = tf.tensor2d(INPUTS); //nested attributes
const INPUTS_TENSOR = tf.tensor1d(INPUTS); //nested attributes
console.log('INPUTS:')
INPUTS_TENSOR.print();
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS); //set of prices
console.log('OUTPUTS:')
OUTPUTS_TENSOR.print();
//NOTE: [0] output correspondes to [0] of inputs.

//NORMALIZE VALUES:
function normalize_model(tensor, min, max){ //saves min max external
    const result = tf.tidy(function(){//auto CLEAN UP of TENSORS, cannot be async
        const MIN_VALUES = min || tf.min(tensor,0);//returns 1D[0,1]
        //zero is axis, can return rows, columns.
        const MAX_VALUES = max || tf.max(tensor,0);//across all values.
        //subtract tensors - every item - zero is MATCH.
        const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
        const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
        // DIVIDE by Range size - as new tensor
        const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);
        return {NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES} //save min max.
        //tidy does not delete these.
    })
    return result;
}
//NORMALIZE inpute features dispose non normalized Tensor
const FEATURE_RESULTS = normalize_model(INPUTS_TENSOR);
console.log('Normalized Values');
FEATURE_RESULTS.NORMALIZED_VALUES.print();
console.log('Min values');
FEATURE_RESULTS.MIN_VALUES.print();
console.log('MaxValues');
FEATURE_RESULTS.MAX_VALUES.print();
INPUTS_TENSOR.dispose();
// OUTPUTS_TENSOR.dispose();

//DEFINE MODEL
const custom_model = tf.sequential(); //flow in sequence.
//2inputs, 1 weight, dense neuron
custom_model.add(tf.layers.dense({inputShape:[1], units:1}));
// custom_model.add(tf.layers.dense({inputShape:[2], units:1}));
custom_model.summary();

function evaluate_model(){
    //predict for single data
    tf.tidy(function(){ //use saved min max
        // let newInput = normalize_model(tf.tensor2d([[750,1]]),FEATURE_RESULTS.MIN_VALUES,FEATURE_RESULTS.MAX_VALUES);
        let newInput = normalize_model(tf.tensor1d([7]),FEATURE_RESULTS.MIN_VALUES,FEATURE_RESULTS.MAX_VALUES);
        //expect 7*7 as result.49
        let output = custom_model.predict(newInput.NORMALIZED_VALUES);
        console.log('TensorAnswer:')
        output.print(); //tensor answer49
    })

    FEATURE_RESULTS.MIN_VALUES.dispose();
    FEATURE_RESULTS.MAX_VALUES.dispose();
    custom_model.dispose();
    console.log(tf.memory().numTensors);

    //SAVE MODEL to COMPUTER
    // await model.save('downloads://my-model');
    //SAVE LOCAL STORAGE
    // await model.save('localstorage://demo/exampleModel1')

    //LOAD SAVED MODEL
    //const model = await tf.loadLayersModel('http://yoursite.com/model.json');
    // LOAD LOCAL storage
    //const model = await tf.loadLayersModel('localstorage://demo/exampleModel1')


}

function logProgress(epoch,logs){
    console.log('Epoch Progress:',epoch,Math.sqrt(logs.loss))//MSE needs sqrt
    if(epoch==70){
        OPTIMIZER.setLearningRate(LEARNING_RATE/2);
    }
}

async function train_model(){ //#10:45, only train once, predict many times.
    // const LEARNING_RATE = 0.01;//steps to change weight / bias.
    // //number too high = NaN , number low slow.
    // const OPTIMIZER = tf.train.sgd(LEARNING_RATE);
    //COMPILE MODEL with LRate and LOSS fn()
    //- compiling freezes model, finalized changes.
    custom_model.compile({
        optimizer : OPTIMIZER,//small data
        // optimizer:tf.train.sgd(LEARNING_RATE),
        //decides weight/bias method, SGD: stochastic gradiesnt descent.
        loss:'meanSquaredError'//MSE: loss. can customize. use sqrt downstrem.
    });
    let results = await custom_model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR, {
        callbacks:{onEpochEnd:logProgress},//SmallData end of every epoch, see progress
        //do not SPLIT for SMALL DATA
        // validationSplit:0.15, //training data percent 15% to train
        shuffle:true,         //data shuffle for input + output (correlates but not in sequence).
        batchSize:2,  //small data sample
        // batchSize:64,  //1 noisy no line of best fit, 64 time sample line fit.
        epochs:80  //no learning past 80
        // epochs:200  //get a chance to learn from small data - longer run.#4:14
        // epochs:10  //go through training once, here 10.
    }) //experiment on these numbers. to find "best line of fit"

    OUTPUTS_TENSOR.dispose();
    FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

    console.log('avg errloss',
        Math.sqrt(results.history.loss[results.history.loss.length-1]));
    // console.log('avg valid new data', //use sqrt because MSE loss
    //     Math.sqrt(results.history.val_loss[results.history.val_loss.length-1]));

    evaluate_model();
}

//******************TRAINING VARIABLES--------------------
const LEARNING_RATE = 0.01;//steps to change weight / bias.
//number too high = NaN , number low slow.
const OPTIMIZER = tf.train.sgd(LEARNING_RATE);

train_model();

