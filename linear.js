let normaliseFeatureTensor, normaliseLabelTensor, trainingFeatureTensor, testingFeatureTensor, trainingLabelTensor,
    testingLabelTensor,model;


const plot = async (pointsArray, featureName) => {
    tfvis.render.scatterplot(
        {name: `${featureName} vs House Price`},
        {values: [pointsArray], series: ['original']},
        {
            xLabel: featureName,
            yLabel: 'Price'
        }
    )
}
const denormalise = (tensor, min, max) => {
    return tensor.mul(max.sub(min)).add(min)
}
const normalise = (tensor) => {
    const min = tensor.min()
    const max = tensor.max()

    return {tensor: tensor.sub(min).div(max.sub(min)), min, max}
}

const createModel = () => {
    model = tf.sequential()
    model.add(tf.layers.dense({
        units: 1,
        useBias: true,
        activation: 'linear',
        inputDim: 1
    }))
    const optimizer = tf.train.sgd(0.1)
    model.compile({
        loss: 'meanSquaredError',
        optimizer
    })
    return model
}
const predict = async () => {
    alert("Not yet implemented");
}
const load = async () => {
    alert("Not yet implemented");
}
const save = async () => {
    alert("Not yet implemented");
}
const test = async () => {
    const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor)
    const loss = await lossTensor.dataSync()
    console.log(`Training set loss: ${loss}`)
    document.getElementById('testing-status').innerHTML=`Testing set loss ${loss}`
}
const train = async () => {
    ['train','test','load','predict','save'].forEach(id=>{
        document.getElementById(`${id}-button`).setAttribute('disabled','disabled')
    });
    document.getElementById('model-status').innerHTML = 'Training.....'
    const model = createModel()
    // model.summary()
    tfvis.show.modelSummary({name: `Model Summary`, tab: `Model`}, model);
    const layer = model.getLayer(undefined, 0)
    tfvis.show.layer({name: "Layer 1"}, layer)
    const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor)
    const trainingLoss = result.history.loss.pop()
    console.log(`Training set loss: ${trainingLoss}`)

    const validationLoss = result.history.val_loss.pop()
    console.log(`Validation set loss: ${validationLoss}`)

    document.getElementById('model-status').innerHTML ="Trained (unsaved)\n"+
        `Loss: ${trainingLoss.toPrecision(5)}\n`
        + `Validation loss: ${validationLoss.toPrecision(5)}`
    document.getElementById('test-button').removeAttribute('disabled')

}
const toggleVisor = async () => {
    tfvis.visor().toggle()
}
const trainModel = async (model, trainingFeatureTensor, trainingLabelTensor) => {
    const {onBatchEnd, onEpochEnd} = tfvis.show.fitCallbacks({
        name: 'Training Performance'
    }, ['loss'])
    return model.fit(trainingFeatureTensor, trainingLabelTensor, {
        batchSize: 32,
        epochs: 20,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd,
        }
    })

}
const run = async () => {
    //Import from csv
    const houseSalesDataset = tf.data.csv("http://127.0.0.1:5500/kc_house_data.csv")

    //Extract x and y values
    const pointsDataset = houseSalesDataset.map(record => ({
        x: record.sqft_living,
        y: record.price
    }))
    const points = await pointsDataset.toArray()
    if (points.length % 2 !== 0) {
        points.pop()
    }
    tf.util.shuffle(points)

    plot(points, 'Square feet')

    //Extract Features (inputs)
    const featureValues = await points.map(p => p.x)
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1])

    //Extract Labels (outputs)

    const labelValues = await points.map(p => p.y)
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1])

//Normalise features and labels
    normaliseFeatureTensor = normalise(featureTensor)
    normaliseLabelTensor = normalise(labelTensor)
    featureTensor.dispose();
    labelTensor.dispose();
    [trainingFeatureTensor, testingFeatureTensor] = tf.split(normaliseFeatureTensor.tensor, 2);
    [trainingLabelTensor, testingLabelTensor] = tf.split(normaliseLabelTensor.tensor, 2);
    trainingFeatureTensor.print(true)

    document.getElementById('model-status').innerHTML = 'No model trained';
    document.getElementById('train-button').removeAttribute('disabled')


    //denormalise(normaliseFeatureTensor.tensor,normaliseFeatureTensor.min,normaliseFeatureTensor.max).print()


}

run()
