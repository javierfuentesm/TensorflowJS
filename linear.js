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
    const model = tf.sequential()
    model.add(tf.layers.dense({
        units: 1,
        useBias: true,
        activation: 'linear',
        inputDim: 1
    }))
    return model
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
    const normaliseFeatureTensor = normalise(featureTensor)
    const normaliseLabelTensor = normalise(labelTensor)

    const [trainingFeatureTensor, testingFeatureTensor] = tf.split(normaliseFeatureTensor.tensor, 2)
    const [trainingLabelTensor, testingLabelTensor] = tf.split(normaliseLabelTensor.tensor, 2)
    trainingFeatureTensor.print(true)
    //denormalise(normaliseFeatureTensor.tensor,normaliseFeatureTensor.min,normaliseFeatureTensor.max).print()

    const model = createModel()
    // model.summary()
    tfvis.show.modelSummary({name: "Model summary"}, model)
    const layer = model.getLayer(undefined, 0)
    tfvis.show.layer({name: "Layer 1"}, layer)
}

run()
