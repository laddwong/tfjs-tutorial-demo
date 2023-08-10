const loadingBox = document.getElementById('loadingBox')
let net;
const classifier = knnClassifier.create();

function getUploadImg(event) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = function(e) {
      const img = document.createElement('img');
      img.onload = function() {
        resolve(img);
      }
      img.src = e.target.result;
    }
    reader.readAsDataURL(event.target.files[0]);
  })
}

function showLoading(msg) {
  loadingBox.innerHTML = msg || 'loading...';
  loadingBox.style.display = 'block';
}

function hideLoading() {
  loadingBox.style.display = 'none';
}

async function doPredict(img) {
  const result = await net.classify(img)
  console.log(result)
  return result
}

async function onUploadImgAndPredict(event) {
  showLoading('predicting...')
  const img = await getUploadImg(event)
  await doPredict(img)
  hideLoading()
}

async function onAddClass(event, classId) {
  showLoading('adding...')
  const img = await getUploadImg(event)
  const activation = net.infer(img, true)
  classifier.addExample(activation, classId)
  hideLoading()
}

async function onUploadClassifyImg(event) {
  showLoading('predicting...')
  const img = await getUploadImg(event)
  const activation = net.infer(img, 'conv-preds')
  const result = await classifier.predictClass(activation)
  console.log(result)
  hideLoading()
  return result
}

async function run () {
  showLoading('loading...')
  net = await mobilenet.load()
  hideLoading()
}

run()