import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.164/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.164/examples/jsm/controls/OrbitControls.js';

const root = document.getElementById('renderRoot');
const annotationBar = document.getElementById('annotationBar');
const learningRateSlider = document.getElementById('learningRate');
const epochSlider = document.getElementById('epochs');
const batchSlider = document.getElementById('batchSize');
const optimizerSelect = document.getElementById('optimizer');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');
const lrValue = document.getElementById('learningRateValue');
const epochValue = document.getElementById('epochValue');
const batchValue = document.getElementById('batchValue');
const autoRotate = document.getElementById('autoRotate');
const showAnnotations = document.getElementById('showAnnotations');
const helpToggle = document.getElementById('helpToggle');
const helpPanel = document.getElementById('helpPanel');
const closeHelp = document.getElementById('closeHelp');

let scene, camera, renderer, controls;
let scatter, boundaryMesh, light;
let lossChart, accChart;
let data, model;
let training = false;
let epochCount = 0;

const renderSizes = {
  width: () => root.clientWidth || root.parentElement.clientWidth,
  height: () => root.clientHeight || 400,
};

function createScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color('#0b1220');

  camera = new THREE.PerspectiveCamera(60, renderSizes.width() / renderSizes.height(), 0.1, 100);
  camera.position.set(2.8, 2.2, 2.8);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.8));
  renderer.setSize(renderSizes.width(), renderSizes.height());
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  root.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = autoRotate.checked;
  controls.autoRotateSpeed = 1.2;

  const ambient = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambient);
  light = new THREE.DirectionalLight(0xffffff, 0.8);
  light.position.set(3, 4, 2);
  scene.add(light);

  const grid = new THREE.GridHelper(4, 8, 0x274060, 0x1f2a44);
  grid.position.y = -1.2;
  scene.add(grid);
}

function generateData(n = 160) {
  const xs = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() - 0.5) * 3;
    const y = (Math.random() - 0.5) * 3;
    const z = (Math.random() - 0.5) * 3;
    const plane = x * 0.8 + y * -0.5 + z * 0.6;
    const label = plane > 0 ? 1 : 0;
    xs.push([x, y, z]);
    ys.push(label);
  }
  return { xs, ys };
}

function createScatter() {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(data.xs.length * 3);
  const colors = new Float32Array(data.xs.length * 3);

  data.xs.forEach((p, i) => {
    positions.set(p, i * 3);
    const color = data.ys[i] === 1 ? new THREE.Color('#48b6ff') : new THREE.Color('#7c3aed');
    colors.set([color.r, color.g, color.b], i * 3);
  });

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({ size: 0.06, vertexColors: true, transparent: true, opacity: 0.95 });
  scatter = new THREE.Points(geometry, material);
  scene.add(scatter);
}

function createDecisionBoundary() {
  const geometry = new THREE.PlaneGeometry(3.5, 3.5, 50, 50);
  geometry.rotateX(-Math.PI / 2);
  const material = new THREE.MeshStandardMaterial({
    color: '#48b6ff',
    transparent: true,
    opacity: 0.25,
    side: THREE.DoubleSide,
    roughness: 0.4,
    metalness: 0.0,
  });
  boundaryMesh = new THREE.Mesh(geometry, material);
  boundaryMesh.position.y = 0;
  scene.add(boundaryMesh);
}

function resizeRenderer() {
  const width = renderSizes.width();
  const height = renderSizes.height();
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function setupCharts() {
  const lossCtx = document.getElementById('lossChart');
  const accCtx = document.getElementById('accChart');
  lossChart = new Chart(lossCtx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Loss', data: [], borderColor: '#48b6ff', backgroundColor: 'rgba(72,182,255,0.2)', tension: 0.2, fill: true, pointRadius: 0 }] },
    options: { scales: { y: { beginAtZero: true } }, plugins: { legend: { display: false } }, animation: false }
  });
  accChart = new Chart(accCtx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Accuracy', data: [], borderColor: '#34d399', backgroundColor: 'rgba(52,211,153,0.2)', tension: 0.2, fill: true, pointRadius: 0 }] },
    options: { scales: { y: { min: 0, max: 1 } }, plugins: { legend: { display: false } }, animation: false }
  });
}

function buildModel() {
  if (model) {
    model.dispose();
  }
  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [3], units: 1, activation: 'sigmoid', kernelInitializer: 'glorotNormal' }));
  const lr = parseFloat(learningRateSlider.value);
  const optimizerChoice = optimizerSelect.value;
  let optimizer;
  if (optimizerChoice === 'adam') optimizer = tf.train.adam(lr);
  else if (optimizerChoice === 'adagrad') optimizer = tf.train.adagrad(lr);
  else optimizer = tf.train.momentum(lr, 0.9, true);

  model.compile({ optimizer, loss: 'binaryCrossentropy', metrics: ['accuracy'] });
}

function tensorData() {
  const xs = tf.tensor2d(data.xs);
  const ys = tf.tensor2d(data.ys, [data.ys.length, 1]);
  return { xs, ys };
}

function updateAnnotation(message) {
  if (showAnnotations.checked) {
    annotationBar.textContent = message;
  } else {
    annotationBar.textContent = '';
  }
}

async function trainModel() {
  if (training) return;
  training = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  resetBtn.disabled = true;
  epochCount = 0;

  const totalEpochs = parseInt(epochSlider.value, 10);
  const batchSize = parseInt(batchSlider.value, 10);
  updateAnnotation('Training kicked off — watching how the boundary reacts...');

  const { xs, ys } = tensorData();

  for (let i = 0; i < totalEpochs && training; i++) {
    const history = await model.fit(xs, ys, {
      epochs: 1,
      batchSize,
      shuffle: true,
    });

    const loss = history.history.loss[0];
    const acc = history.history.acc[0] ?? history.history.accuracy[0];
    lossChart.data.labels.push(`e${epochCount + 1}`);
    accChart.data.labels.push(`e${epochCount + 1}`);
    lossChart.data.datasets[0].data.push(loss);
    accChart.data.datasets[0].data.push(acc);
    lossChart.update();
    accChart.update();

    epochCount++;
    updateBoundaryGeometry();
    updatePointColors();

    updateAnnotation(`Epoch ${epochCount}: loss ${loss.toFixed(3)}, accuracy ${(acc * 100).toFixed(1)}%`);

    await tf.nextFrame();
  }

  xs.dispose();
  ys.dispose();
  training = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  resetBtn.disabled = false;
  updateAnnotation(training ? 'Training in progress...' : 'Training stopped or finished. You can tweak parameters and try again.');
}

function stopTraining() {
  training = false;
  updateAnnotation('Training interrupted. Adjust parameters or resume.');
}

function reset() {
  training = false;
  epochCount = 0;
  lossChart.data.labels = [];
  lossChart.data.datasets[0].data = [];
  accChart.data.labels = [];
  accChart.data.datasets[0].data = [];
  lossChart.update();
  accChart.update();

  data = generateData();
  buildModel();
  updateScatterPositions();
  updateBoundaryGeometry(true);
  updateAnnotation('New dataset sampled — notice the scatter distribution changed.');
}

function updateScatterPositions() {
  if (!scatter) return;
  const positions = scatter.geometry.getAttribute('position');
  const colors = scatter.geometry.getAttribute('color');
  data.xs.forEach((p, i) => {
    positions.setXYZ(i, p[0], p[1], p[2]);
    const color = data.ys[i] === 1 ? new THREE.Color('#48b6ff') : new THREE.Color('#7c3aed');
    colors.setXYZ(i, color.r, color.g, color.b);
  });
  positions.needsUpdate = true;
  colors.needsUpdate = true;
}

function updateBoundaryGeometry(reset = false) {
  if (!boundaryMesh) return;
  const positionAttr = boundaryMesh.geometry.getAttribute('position');
  const weights = model.getWeights();
  if (!weights.length) return;
  const kernel = weights[0].arraySync();
  const bias = weights[1].arraySync();
  const w1 = kernel[0][0];
  const w2 = kernel[1][0];
  const w3 = kernel[2][0];
  const b = bias[0];
  const safeW3 = Math.abs(w3) < 1e-4 ? 1e-4 : w3;
  for (let i = 0; i < positionAttr.count; i++) {
    const x = positionAttr.getX(i);
    const z = positionAttr.getZ(i);
    const y = reset ? 0 : -(w1 * x + w2 * z + b) / safeW3;
    positionAttr.setY(i, y);
  }
  positionAttr.needsUpdate = true;
  boundaryMesh.geometry.computeVertexNormals();
}

function updatePointColors() {
  const colors = scatter.geometry.getAttribute('color');
  tf.tidy(() => {
    const preds = model.predict(tf.tensor2d(data.xs)).dataSync();
    preds.forEach((p, i) => {
      const predictedClass = p > 0.5 ? 1 : 0;
      const base = predictedClass === data.ys[i] ? '#34d399' : '#f87171';
      const color = new THREE.Color(base);
      colors.setXYZ(i, color.r, color.g, color.b);
    });
  });
  colors.needsUpdate = true;
}

function animate() {
  requestAnimationFrame(animate);
  controls.autoRotate = autoRotate.checked;
  controls.update();
  renderer.render(scene, camera);
}

function init() {
  createScene();
  data = generateData();
  createScatter();
  createDecisionBoundary();
  buildModel();
  setupCharts();
  updateBoundaryGeometry(true);
  animate();

  stopBtn.disabled = true;

  window.addEventListener('resize', resizeRenderer);
  resizeRenderer();

  learningRateSlider.addEventListener('input', () => {
    lrValue.textContent = parseFloat(learningRateSlider.value).toFixed(4);
    buildModel();
  });
  epochSlider.addEventListener('input', () => (epochValue.textContent = epochSlider.value));
  batchSlider.addEventListener('input', () => (batchValue.textContent = batchSlider.value));
  optimizerSelect.addEventListener('change', () => buildModel());

  startBtn.addEventListener('click', trainModel);
  stopBtn.addEventListener('click', stopTraining);
  resetBtn.addEventListener('click', reset);

  showAnnotations.addEventListener('change', () => {
    annotationBar.classList.toggle('muted', !showAnnotations.checked);
    annotationBar.textContent = showAnnotations.checked ? 'Annotations enabled — watch for guidance here.' : '';
  });

  helpToggle.addEventListener('click', () => helpPanel.classList.add('open'));
  closeHelp.addEventListener('click', () => helpPanel.classList.remove('open'));

  updateAnnotation('Ready. Pick hyperparameters and start training to see the boundary move.');
}

init();
