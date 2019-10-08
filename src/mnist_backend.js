//"use strict";

import * as tf from '@tensorflow/tfjs';
import tfserialize from './tfserialize';


//coordinates the detections on the provided batch of images
async function computeBatch(
    myImages, myShapes, testSerialize = true, layersModel = false
) {
    //compute a batch of images using the following protocol ("Useful for debugging")
    //var myCompute = 'browser';
    var myCompute = 'scheduler';

    //Load the model first (Either layers or graph) useful for testing the same model but in different tfjs formats
    let myModel = null;
    if (layersModel) {
        myModel = await tf.loadLayersModel("http://127.0.0.1:5500/layers-js/model.json");
    } else {
        myModel = await tf.loadGraphModel("http://127.0.0.1:5500/graph-js/model.json");
    }

    let myResults = [];

    if (myCompute == 'local' || myCompute == 'scheduler') {

        //DCP computation prompts user for a wallet, then passes on the images
        await dcp.protocol.keychain.getKeystore();
        //call dcp with the images, the model, the boolean which asks if this is a layers model and the compute format
        myResults = await _callDCP(myImages, myModel, layersModel, myCompute);

    } else if (myCompute == 'browser') {

        if (testSerialize && layersModel) {
            console.log("Layers Model is being Serialized");
            let ser_model = await tfserialize.serialize(myModel, false);
            console.log("Model size after serialization is: ", Buffer.byteLength(ser_model, 'utf8'));
            myModel = await tfserialize.deserialize(ser_model);
            console.log("Model has been deserialized and replaced");
        } else if (testSerialize && !layersModel) {
            console.log("Graph Model is being Serialized");
            let ser_model = await tfserialize.serializeGraph(myModel);
            console.log("Model size after serialization is: ", Buffer.byteLength(ser_model, 'utf8'));
            myModel = await tfserialize.deserializeGraph(ser_model);
            console.log("Model has been deserialized and replaced");
        }


        for (let i = 0; i < myImages.length; i++) {

            //converts images from float32array to regular array
            let thisImage = myImages[i];
            let thisShape = myShapes[i];

            //converts image array to an appropriate tensor prior to loading into the model
            tf.setBackend('webgl');
            let imageTensor = await tf.tidy(() => {
                let myTensor = tf.tensor(thisImage, thisShape);
                return myTensor.expandDims(0).toFloat().div(255.);
            });

            //calls loaded object detection model with the provided image tensor and adds the results to an array
            let thisPrediction = await myModel.predict(imageTensor);
            let predictions = await tf.argMax(thisPrediction, -1).data();
            //      let thisPrediction = await global.model.detectFromTensor(imageTensor);

            //prediction tensor gives boxes between 0 and 480, need to normalize in future
            imageTensor.dispose();
            thisPrediction.dispose();
            myResults.push(predictions[0]);
        }
    }

    return myResults;
}

//function takes an array of [num_images] and turns it into [[batch_size]*(min(1,num_images//batch_size))] 
// if there are less than batchSize images, we simply return [[num_images]]
async function batchify(allImages, batchSize) {
    let batchified = []
    let curBatch = []
    for (let i = 0; i < allImages.length; i++) {
        curBatch.push(allImages[i]);
        if (curBatch.length >= batchSize) {
            batchified.push(curBatch);
            curBatch = [];
        }
    }
    if (curBatch.length >= 1) {
        batchified.push(curBatch);
    }
    return batchified;
}

//distributed computation of frame batches (requires an available DCP network)
async function _callDCP(
    myImages,
    myModel,
    layersModel,
    myCompute,
    batchSize = 4,
) {

    //begin by serializing model
    let ser_model = null;

    if (layersModel) {
        console.log("Layers Model is being Serialized");
        ser_model = await tfserialize.serialize(myModel, false);
        console.log("Model size after serialization is: ", Buffer.byteLength(ser_model, 'utf8'));
    } else if (!layersModel) {
        console.log("Graph Model is being Serialized");
        ser_model = await tfserialize.serializeGraph(myModel);
        console.log("Model size after serialization is: ", Buffer.byteLength(ser_model, 'utf8'));
    }
    //set myModel to null for gc
    myModel = null;


    let myResults = [];

    //(here I am just testing out some batch stuff so I push the same images on a second time.)
    myImages.push(myImages[0]);

    //batchify the images
    myImages = await batchify(myImages, batchSize);


    //stringify the image batches
    for (let i = 0; i < myImages.length; i++) {
        myImages[i] = JSON.stringify(myImages[i]);
    }

    //the inference Object contains all we need for inference (in the case of mnist)
    let inferenceObj = { serialModel: ser_model, layersModel: layersModel };


    //the actual work function being run on the other side
    const workFunction = `async function (images, inferenceObj) {
    //load necessary DCP modules
    tf = require('tfjs');
    serialize = require('serialize');
    tfserialize = require('tfserialize');


    progress(1);


    return inferenceObj;
        
  }`;

    //the generator for dcp compute protocol: myImages will be sliced out and sent to different workers, 
    //while inferenceObj will be sent to each of the workers in it's entirety.
    const gen = dcp.compute.for(myImages, workFunction, [inferenceObj]);

    //define necessary modules, which must be available on the DCP module server 
    gen.requires('dcp-core/serialize')
    gen.requires('tensorflowdcp/tfjs');
    gen.requires('modelserial/tfserialize.js');

    // gen.on('accepted', (thisOutput) => {
    //     console.log('Job accepted!');
    //     console.log(gen.id);
    // });

    // gen.on('complete', (thisOutput) => {
    //     console.log('Job complete!');
    // });

    //receives and collects the results of each worker thread
    gen.on('result', (thisOutput) => {
        myResults.push(thisOutput.result);
    });

    //worker thread title on DCP
    gen._generator.public = {
        name: 'Test'
    };

    if (myCompute == 'scheduler') {
        //sets gpu requirement and DCC payment, calls DCP generator on scheduler server
        //gen._generator.capabilities = {gpu: true};
        //this is the actual call, the 0.0001 is the amount we are willing to pay per slice
        await gen.exec(0.0001);

        // the results
        console.log('inference Object: ', myResults);
        //check if string model is the same as the one before we sent (making sure nothing is being corrupted)
        console.log('are the models the same?: ', inferenceObj.serialModel === myResults[0].serialModel);
        throw "Here";
        return myResults;
    } else if (myCompute == 'local') {
        //sets number of local cores, calls DCP generator on the local machine
        let workerNumber = config.computeWorkers;
        await gen.localExec(workerNumber);
        return myResults;
    }
}



const platform = {
    computeBatch
};

export default platform;