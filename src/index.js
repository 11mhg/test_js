// prestriate demo application
// aisight inc
// this build is not for commercial use

//"use strict";
import * as tf from '@tensorflow/tfjs'
import backend from './backend';

let globalImageArray = null;
let globalImageShape = null;

console.log("Beginning Page");

document.querySelector('input[type="file"]').value = "";



window.addEventListener('load', function() {
    this.document.querySelector('input[type="file"]').addEventListener('change', function() {
        if (this.files && this.files[0]) {
            let image = new Image();
            image.src = URL.createObjectURL(this.files[0]);
            image.onload = imageIsLoaded;
        }
    })
})

function aspectRatioResize(width, height, maxWidth, maxHeight) {
    let ratio = 0;

    if (width > maxWidth) {
        ratio = maxWidth / width;
        height = height * ratio;
        width = width * ratio;
    }

    if (height > maxHeight) {
        ratio = maxHeight / height;
        height = height * ratio;
        width = width * ratio;
    }
    return [width, height];
}

async function imageIsLoaded(e) {
    let currentImage = e.target;
    var canvas = document.getElementById('original_canvas');
    let new_sizes = aspectRatioResize(currentImage.naturalWidth, currentImage.naturalHeight, 800, 600);
    canvas.width = currentImage.naturalWidth;
    canvas.height = currentImage.naturalHeight;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(currentImage, 0, 0, currentImage.naturalWidth,currentImage.naturalHeight);

    let pixelData = ctx.getImageData(0, 0, currentImage.naturalWidth, currentImage.naturalHeight);
    let pixelTensor = await tf.browser.fromPixels(pixelData);

    canvas.width = new_sizes[0];
    canvas.height = new_sizes[1];
    ctx.drawImage(currentImage,0,0, new_sizes[0], new_sizes[1]);

    globalImageArray = new Float32Array((await pixelTensor.data()));
    globalImageShape = pixelTensor.shape;

    var canvas = document.getElementById('predicted_canvas');
    canvas.width = new_sizes[0];
    canvas.height = new_sizes[1];
    var ctx = canvas.getContext('2d');
    ctx.drawImage(currentImage, 0, 0, new_sizes[0], new_sizes[1]);
}

function f32ToCanvas(pixeldata,pixelshape){
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');

    ctx.width = pixelshape[1];
    ctx.height = pixelshape[0];

    var imageData = ctx.createImageData(pixelshape[1],pixelshape[0]);
    var data = imageData.data;
    var len = data.length;
    var i=0;
    var t=0;

    for (;i<len;i+=4){
        data[i] = pixeldata[t];
        data[i+1] = pixeldata[t+1];
        data[i+2] = pixeldata[t+2];
        data[i+3] = 255;
        t+=3;
    }
    ctx.putImageData(imageData,0,0);
    return canvas;
}


//helper function which resizes pixels data to a particular width and height.
async function resizepixelarray(
    pixeldata, pixelshape, resize_width, resize_height,keep_aspect_ratio=false) {
    var canvas = f32ToCanvas(pixeldata,pixelshape);

    var newCanvas = document.createElement('canvas');
    if (keep_aspect_ratio){
        let new_size = aspectRatioResize(pixelshape[1],pixelshape[0],resize_width,resize_height);
        resize_width = new_size[0];
        resize_height = new_size[1];
    }

    newCanvas.width = resize_width;
    newCanvas.height = resize_height;
    var newctx = newCanvas.getContext("2d");
    newctx.drawImage(canvas,0,0);

    let newData = newctx.getImageData(0, 0, resize_width, resize_height);
    let pixelTensor = await tf.browser.fromPixels(newData);


    let retData = new Float32Array((await pixelTensor.data()));
    let retShape = pixelTensor.shape;

    return [retData, retShape];
}


async function infer() {
    if (document.getElementById('uploaded_image').value.length < 4) {
        alert("Select photo for upload");
        return false;
    }

    var select = document.getElementById("model-type");
    var selectedVal = parseInt(select.options[select.selectedIndex].value);

    console.log(selectedVal);

    if (selectedVal == 1) {
        let resize_ret = await resizepixelarray(globalImageArray, globalImageShape, 480, 480);
        globalImageArray = [resize_ret[0]];
        globalImageShape = [resize_ret[1]];

        let resultBoxes = await backend.computeBatch(globalImageArray,globalImageShape);
        for (let i=0;i<resultBoxes.length;i++){
            let bb = resultBoxes[i];
            await backend.drawBoxestoContext( bb, document.getElementById('predicted_canvas').getContext("2d"));
        }
    }

    console.log("Done Inference");

    return true;
}



document.getElementById("infer").onclick = infer;
