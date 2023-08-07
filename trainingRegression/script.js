
async function getData() {
  const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
  const carsData = await carsDataResponse.json();
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  }))
  .filter(car => (car.mpg != null && car.horsepower != null));
  console.log('获取原始数据：', JSON.parse(JSON.stringify(cleaned)))
  return cleaned;
}


async function run() {
  // 使用tfvis显示原始的数据
  const data = await getData();
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    {name: 'Horsepower v MPG'},
    {values},
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );

  // 创建模型实例，并在tfvis中显示
  const model = createModel()
  tfvis.show.modelSummary({name: 'Model Summary'}, model)

  // 数据处理
  const tensorData = convertData(data)

  // 训练模型
  await trainModel(model, tensorData.normalizedInputs, tensorData.normailzedLabels)

  // 测试模型
  testModel(model, data, tensorData)
}

function createModel() {
  const model = tf.sequential();// 实例化序贯模型
  // 给模型添加输入层，dense表示的是密集层，units是输出空间的维度（理解成神经元的数量）
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true , activation: 'sigmoid'}))
  // 添加输出层



  return model
}

function convertData(data) {
  return tf.tidy(() => {
    // 打乱顺序
    tf.util.shuffle(data)
    // 张量转换
    const inputs = data.map(item => item.horsepower)
    const labels = data.map(item => item.mpg)
    const inputsTensor = tf.tensor2d(inputs, [inputs.length, 1])
    const labelsTensor = tf.tensor2d(labels, [labels.length, 1])

    // 最大最小值，用来归一化和还原
    const inputMax = inputsTensor.max();
    const inputMin = inputsTensor.min();
    const labelMax = labelsTensor.max();
    const labelMin = labelsTensor.min();

    // 归一化
    const normalizedInputs = inputsTensor.sub(inputMin).div(inputMax.sub(inputMin))
    const normailzedLabels = labelsTensor.sub(labelMin).div(labelMax.sub(labelMin))

    return {
      normalizedInputs,// 归一化后的输入
      normailzedLabels,// 归一化后的答案
      inputMax,// 最大输入值
      inputMin,// 最小输入值
      labelMax,// 最大答案
      labelMin// 最小答案
    }
  })
}

async function trainModel(model, inputs, labels) {
  // 编译模型
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse']
  })

  // 开始训练，每次周期结束，在tfvis中显示指标
  const batchSize = 32
  const epochs = 500
  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      {name: 'Tranining Performance'},
      ['loss', 'mse'],
      {height: 200, callbacks: ['onEpochEnd']}
    )
  })
}

function testModel(model, originData, tensorData) {
  const [testHP, resultMpg] = tf.tidy(() => {
    // 生成测试的样本，马力从0到1，一共一百个（简单粗暴）
    const testHP = tf.linspace(0, 1, 100)
    const resultMp = model.predict(testHP.reshape([100, 1]))

    // 归一化还原
    const unNormTestHP = testHP.mul(tensorData.inputMax.sub(tensorData.inputMin)).add(tensorData.inputMin)
    const unNormResultMp = resultMp.mul(tensorData.labelMax.sub(tensorData.labelMin)).add(tensorData.labelMin)

    return [unNormTestHP.dataSync(), unNormResultMp.dataSync()]
  })

  // 画图，比对原始数据
  const predictedPoints = Array.from(testHP).map((val, i) => {
    return {x: val, y: resultMpg[i]}
  });

  const originalPoints = originData.map(d => ({
    x: d.horsepower, y: d.mpg,
  }));

  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'},
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
}

run()