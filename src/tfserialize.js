'use strict';

import * as tf from '@tensorflow/tfjs';
import serialjs from './serialize';



/**
 * A tf.IOHandler that's only purpose is to override the tf.graph IO handler 
 */
class GraphIOHandler{
    constructor(modelArtifacts){
        this.modelArtifacts = modelArtifacts;
    }

    async load(){
        return this.modelArtifacts;
    }
}


/**
 * A tf.IOHandler that serializes a model into a string.
 */
class SerializeIOHandler {

    /**
     * @param {string} model - the serialized model, if loading. Othersize, the
     * serialized model will go in this.model after saving is complete.
     */
    constructor(model) {
        this.model = model;
    }

    /**
     * Serializes the model.
     * @param {tf.IOHandler} modelArtifacts - The weights and topology of the model.
     * @return {Promise<tf.SaveResult>} - A promise resolving to an object containing error information if applicable. Not often used.
     */
    async save(modelArtifacts) {
        this.model = serialjs.serialize(modelArtifacts);
        return {
            modelArtifactsInfo: {
                dateSaved: new Date(),
                // Note that this isn't exactly correct, since we're not using
                // JSON, but rather serialize.js for this.
                modelTopologyType: 'JSON',
                modelTopologyBytes: modelArtifacts.modelTopology == null ?
                    0 : JSON.stringify(modelArtifacts.modelTopology).length,
                weightSpecsBytes: modelArtifacts.weightSpecs == null ?
                    0 : JSON.stringify(modelArtifacts.weightSpecs).length,
                weightDataBytes: modelArtifacts.weightData == null ?
                    0 : modelArtifacts.weightData.byteLength,
            }
        };
    }

    /**
     * Deserializes the model using this.model.
     */
    async load() {

        if (typeof this.model === "undefined") {
            throw new Error("SerializeIOHandler.load() called without providing a model to load.");
        }

        return serialjs.deserialize(this.model);
    }
}

/**
 *This function serializes the training information that is needed to train the model
 */
async function getTrainingInfo(model) {
    //model.optimizer.getClassName() returns the class name of the optimizer, so "Adam", or "SGD",
    // it is needed to find the constructor of the optimizer suring deserialization in the worker.
    //model.optimizer.getConfig() returns all the stateful information of the optimizer
    const obj = {
        optimizerName: await model.optimizer.getClassName(),
        optimizerConfig: await model.optimizer.getConfig(),
        loss: model.loss,
        metrics: model.metrics
    }

    return await JSON.stringify(obj)
}

/**
 *This function deserializes the training information that is needed to train the model
 */
async function fromTrainingInfo(trainingInfoJSON) {
    const trainingInfo = await JSON.parse(trainingInfoJSON)

    const className = trainingInfo.optimizerName
    const config = trainingInfo.optimizerConfig

    //tf.serialization.SerializationMap.getMap().classNameMap returns the class name map that tensorflow uses
    //this object maps class names, eg "Adam" or "SGD", to an array of two objects. The first object is the 
    //constructor of the corresponding class, the second object is a function that parses the classes config
    //object (the thing that holds all its stateful information) into the constructor, and returns an instance
    //of the class. If you define your own optimizer and want to use it with this method, you must add that
    //optimizer to the serialization map using the registerClass function from tensorflow's serialization
    //library (not the serialization library Wes wrote).
    const temp = tf.serialization.SerializationMap.getMap().classNameMap[className]

    const constructor = temp[0]
    const parser = temp[1]

    return { optimizer: parser(constructor, config), loss: trainingInfo.loss, metrics: trainingInfo.metrics }
}

/**
 *  Serializes a Tensorflow Graph Model. Note graph models
 *  can only be created from a tensorflow [Saved Model] in python
 *  that has been converted to js using the tensorflowjs_converter
 * @param {tf.GraphModel} - The Graph model that we'd like to serialize. 
 * @returns {string} - The serialized Model
 */
async function serializeGraph(model){
    const modelArtifacts = await model.handler.load();

    const result =  {
        modelArtifactsInfo: {
            dateSaved: new Date(),
            // Note that this isn't exactly correct, since we're not using
            // JSON, but rather serialize.js for this.
            modelTopologyType: 'JSON',
            modelTopologyBytes: modelArtifacts.modelTopology == null ?
                0 : JSON.stringify(modelArtifacts.modelTopology).length,
            weightSpecsBytes: modelArtifacts.weightSpecs == null ?
                0 : JSON.stringify(modelArtifacts.weightSpecs).length,
            weightDataBytes: modelArtifacts.weightData == null ?
                0 : modelArtifacts.weightData.byteLength,
        }
    };
    let serializedArtifacts = serialjs.serialize(modelArtifacts);
    const completeModel = { model: serializedArtifacts };
    return await JSON.stringify(completeModel);
}



/**
 * Deserializes a graph model by creating a 
 * tf.GraphModel and overriding the IO Handler so that
 * all we need to do is to load the graph model in order
 * to override the proper weights and model topology
 * To see what tf.GraphModel looks like look at tensorflow/tfjs-converter/src/executor/graph_model.ts
 * @param {string} - The serialized model
 * @returns {tf.GraphModel} - The graph model
 */
async function deserializeGraph(str){
    const completeModel = await JSON.parse(str);

    const modelArtifacts = serialjs.deserialize(completeModel.model);
    const graphModel = new tf.GraphModel();
    graphModel.modelArtifacts = modelArtifacts;

    graphModel.findIOHandler = function() {
        this.handler = new GraphIOHandler(this.modelArtifacts);
    }

    await graphModel.load();

    return graphModel;

}



/**
 * Serializes a TensorFlow model, turning it into a string.
 * @param {tf.LayersModel} - A model to serialize
 * @param {boolean} - Should we serialize the training info too?
 * @returns {string} - The serialized model.
 */
async function serialize(model,withTrainingInfo) {
    const handler = new SerializeIOHandler();
    const result = await model.save(handler);

    let completeModel = null;

    if (withTrainingInfo){
        try{
            completeModel = { model: handler.model, trainingInfo: await getTrainingInfo(model), withTrainingInfo: true};
        }catch(error){
            console.error(error);
            throw "Model does not have training info!";
        }
    }else{
        completeModel = { model: handler.model, withTrainingInfo: false};
    }

    return await JSON.stringify(completeModel)
}

/**
 * Turns a string back into a TensorFlow model.
 * @param {string} str - The string representing the model
 * @returns {tf.LayersModel} - The original model.
 */
async function deserialize(str) {
    const completeModel = await JSON.parse(str)

    const serializedModel = completeModel.model
    const hasTrainingInfo = completeModel.withTrainingInfo

    const handler = new SerializeIOHandler(serializedModel);
    const model = await tf.loadLayersModel(handler);


    if (hasTrainingInfo){
        const serializedTrainingInfo = completeModel.trainingInfo
        const trainingInfo = await fromTrainingInfo(serializedTrainingInfo)
        model.compile(trainingInfo)
    }


    return model
}

// const ENVIRONMENT_IS_NODE = typeof require === 'function' &&
//     typeof global === 'object' &&
//     typeof global.process === 'object' && typeof global.process.constructor === 'function' &&
//     typeof global.process.release === 'object' && global.process.release.name === 'node';
// const ENVIRONMENT_IS_WORKER = typeof importScripts === 'function';
// const ENVIRONMENT_IS_WEB = typeof window === 'object' && typeof document === 'object' && !ENVIRONMENT_IS_WORKER;
// const isStrict = (function() { return !this; })();

// if (ENVIRONMENT_IS_WEB) {
//     // In the browser, just script tag this in after tensorflow.
//     // Use the tfserialize object this defines.

//     // Browser code (expected to be in the same directory as serialize.js)
//     async function import_serialize() {
//         // Using fetch(./serialize.js) fails if this script isn't located in the root
//         // of the web server.
//         const path_here = document.currentScript.src;
//         const parent_path = path_here.slice(0, path_here.lastIndexOf('/'));
//         return eval(await (await fetch(parent_path + "/serialize.js")).text())
//     }
//     // Leave it as a promise for save and load to await.
//     serialjs = import_serialize();
//     _local_tf = require('@tensorflow/tfjs'); //tf;
//     // Put the code into the tfserialize object.
//     tfserialize = {
//         serialize,
//         deserialize
//     };
//     if (isStrict){
//         serialize = null;
//         deserialize = null;
//     }else{
//         eval('delete serialize;delete deserialize');
//     }
// } else if (ENVIRONMENT_IS_NODE) {
//     // Running in node
//     serialjs = require('./serialize');
//     _local_tf = require('@tensorflow/tfjs');
//     exports.serialize = serialize;
//     exports.deserialize = deserialize;
// } else {
//     // Running in a worker
//     eval("serialjs = require('serialize'); _local_tf = require('tfjs');exports.serialize = serialize;exports.deserialize = deserialize;");
// }

const tfserialize = {
    serialize,
    deserialize,
    serializeGraph,
    deserializeGraph
};

export default tfserialize;