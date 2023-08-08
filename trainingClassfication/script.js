import {MnistData} from './data.js';

const IMG_WIDTH = 28
const IMG_HEIGHT = 28
const IMG_CHANNAL = 1
const CLASS_NAMES = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

async function showExamples(data) {
  // tfvis中创建容器
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});

  // 获取20张样本
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  // 遍历样本，渲染到canvas
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]); // 28×28像素的灰度图片，只有一个颜色通道，所以张量形状是[28, 28, 1]
    });

    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

function createModel() {
  const model = tf.sequential()
  const NUM_OUTPUT_CLASSES = 10;

  model.add(tf.layers.conv2d({
    inputShape: [IMG_WIDTH, IMG_HEIGHT, IMG_CHANNAL], // 输入层必填形状，后面的可以省略写形状
    kernelSize: 5, // 卷积核边长
    filters: 8, // 卷积核数量（也是输出的维度）
    strides: 1, // 步长
    activation: 'relu', // 激活函数
    kernelInitializer: 'varianceScaling' // 用于随机初始化权重
  }))

  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]})) // 池化层或者采样层

  model.add(tf.layers.conv2d({
    kernelSize: 5, 
    filters: 16,
    strides: 1, 
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }))

  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}))

  model.add(tf.layers.flatten()); // 展平层：将多维特征数据展开为一维向量

  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));

  const optimizer = tf.train.adam();
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })
  return model
}

async function trainModel(model, data) {
  const BATCH_SIZE = 512
  const TRAIN_DATA_SIZE = 5500
  const TEST_DATA_SIZE = 1000
  const EPOCHS = 10

  // 训练数据集
  const  [trainImg, trainLabel] = tf.tidy(() => {
    const trainData = data.nextTrainBatch(TRAIN_DATA_SIZE)
    return [
      trainData.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      trainData.labels
    ]
  })

  // 校验数据集
  const  [testImg, testLabel] = tf.tidy(() => {
    const testData = data.nextTrainBatch(TEST_DATA_SIZE)
    return [
      testData.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      testData.labels
    ]
  })

  return model.fit(
    trainImg, 
    trainLabel, 
    {
      batchSize: BATCH_SIZE,
      validationData: [testImg, testLabel],
      epochs: EPOCHS,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        {name: 'Model Training', tab: 'Model', styles: {height: '1000px'}},
        ['loss', 'val_loss', 'acc', 'val_acc']
      )
    }
  )
}

function doPrediction(model, data, testDataSize = 500) {
  const testData = data.nextTestBatch(testDataSize)
  const testImg = testData.xs.reshape([testDataSize, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNAL])
  const testLabel = testData.labels.argMax(-1)
  const result = model.predict(testImg).argMax(-1)

  testImg.dispose()
  return [result, testLabel]
}

async function showConfusion(result, answer) {
  const confusionMatrix = await tfvis.metrics.confusionMatrix(answer, result)
  const container = {name: 'confusion matrix', tab: 'Evaluation'}
  tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: CLASS_NAMES})

  answer.dispose()
}

async function run() {
  // 数据
  const data = new MnistData();
  await data.load();
  await showExamples(data);

  // 创建模型和训练
  const model = createModel()
  tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
  await trainModel(model, data)

  // 模型评估
  const [result, answer] = doPrediction(model, data)
  showConfusion(result, answer)
}

run()