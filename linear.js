const plot = async(pointsArray, featureName)=>{
    tfvis.render.scatterplot(
        { name: `${featureName} vs House Price`},
        { values: [pointsArray], series:['original']},
        {
            xLabel: featureName,
            yLabel:'Price'
        }
    )
}
const denormalise =(tensor,min,max)=>{
    return tensor.mul(max.sub(min)).add(min)
}
const normalise = (tensor)=>{
    const min = tensor.min()
    const max = tensor.max()

    return {tensor:tensor.sub(min).div(max.sub(min)),min,max}


}
const run = async () => {
    const houseSalesDataset = tf.data.csv("http://127.0.0.1:5500/kc_house_data.csv")
    const sampleDataset = houseSalesDataset.take(10)
    const dataArray = await sampleDataset.toArray()
    const points = houseSalesDataset.map(record => ({
        x: record.sqft_living,
        y: record.price
    }))
    plot(await points.toArray(),'Square feet' )

    //Features (inputs)
    const featureValues = await points.map(p=>p.x).toArray()
    const featureTensor = tf.tensor2d(featureValues,[featureValues.length,1])

    const labelValues = await points.map(p=>p.y).toArray()
    const labelTensor = tf.tensor2d(labelValues,[labelValues.length,1])

    featureTensor.print()
    labelTensor.print()

    const normaliseFeatureTensor = normalise(featureTensor)
    const normaliseLabelTensor = normalise(labelTensor)
    normaliseFeatureTensor.tensor.print()
    normaliseLabelTensor.tensor.print()

    denormalise(normaliseFeatureTensor.tensor,normaliseFeatureTensor.min,normaliseFeatureTensor.max).print()
}

run()
