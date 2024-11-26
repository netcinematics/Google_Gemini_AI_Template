import {TRAINING_DATA} from 
'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js'
//Input feature pairs (size, num, view area, zip)
const INPUTS = TRAINING_DATA.inputs; //FEATURES
// use attributes to predict price
const OUTPUTS = TRAINING_DATA.outputs; //PREDICTION VALUES
//shuffle - avoids INPUT PATTERNS. of Data Sequence.
tf.util.shuffleCombo(INPUTS, OUTPUTS);
const INPUTS_TENSOR = tf.tensor2d(INPUTS); //nested attributes
console.log('INPUTS:')
INPUTS_TENSOR.print();
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS); //set of prices
console.log('OUTPUTS:')
OUTPUTS_TENSOR.print();
//NOTE: [0] output correspondes to [0] of inputs.
//NORMALIZE VALUES:
function normalize(tensor, min, max){ //saves min max external
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
const FEATURE_RESULTS = normalize(INPUTS_TENSOR);
console.log('Normalized Values');
FEATURE_RESULTS.NORMALIZED_VALUES.print();
console.log('Min values');
FEATURE_RESULTS.MIN_VALUES.print();
console.log('MaxValues');
FEATURE_RESULTS.MAX_VALUES.print();
INPUTS_TENSOR.dispose();
OUTPUTS_TENSOR.dispose();