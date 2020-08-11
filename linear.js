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

const run = async () => {
    const houseSalesDataset = tf.data.csv("http://127.0.0.1:5500/kc_house_data.csv")
    const sampleDataset = houseSalesDataset.take(10)
    const dataArray = await sampleDataset.toArray()
    const points = houseSalesDataset.map(record => ({
        x: record.sqft_living,
        y: record.price
    }))
    plot(await points.toArray(),'Square feet' )
}

run()
