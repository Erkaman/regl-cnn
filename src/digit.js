var loadData = require('./common.js').loadData

var canvas = document.body.appendChild(document.createElement('canvas'))
canvas.width = 1
canvas.height = 1

const regl = require('regl')({
  canvas: canvas,
  extensions: ['oes_texture_float'],
  onDone: function (err, regl) {
    if (err) {
      document.body.innerHTML =
        'Failed to initialize the demo because:</br></br><code>' + err + '</code></br></br>'
      throw err
    }
  }
})

loadData(function (data) {
  var cnnGpu = require('./gpu.js')(regl, data)

  // container for everything
  var container = document.createElement('div')
  container.style.cssText = 'margin: 0 auto; max-width: 860px;' // center text.
  container.style.fontWeight = '300' // default font weight
  container.style.fontSize = '1.0em' // default font size
  container.style.lineHeight = '1.6em' // default line height
  container.style.fontFamily = "'Roboto',Helvetica,sans-serif"
  container.style.color = '#393939'
  document.body.appendChild(container)

  // h1 heder
  var par = document.createElement('h1')
  par.innerHTML = 'GPU Deep Learning Demo'
  par.style.fontWeight = '400'
  par.style.fontSize = '2em'
  container.appendChild(par)

  // paragraph
  par = document.createElement('p')
  par.innerHTML = [
    'Please draw a character into this canvas.<br>',
    '(It will probably only work in Firefox and Chrome. And it may not work on mobile. It should look like <a href="google.com">this</a>'
  ].join('\n')
  container.appendChild(par)

  // create drawing canvas.
  canvas = document.createElement('canvas')
  canvas.width = 280
  canvas.height = 280
  canvas.style.borderWidth = 1
  canvas.style.borderStyle = 'solid'
  container.appendChild(canvas)
  var ctx = canvas.getContext('2d')

  // digit display div
  var digitPar = document.createElement('div')
  digitPar.innerHTML = ''
  digitPar.style.fontWeight = '800'
  digitPar.style.fontSize = '14em'
  digitPar.style.display = 'inline'
  digitPar.style.marginLeft = '50px'
  container.appendChild(digitPar)

  var btnDiv = document.createElement('div')

  function createBtn () {
    var btn = document.createElement('button')
    btn.style.margin = '3px'
    btnDiv.appendChild(btn)

    return btn
  }

  // create buttons
  var btn = createBtn()
  btn.innerHTML = 'Clear'
  btn.addEventListener('click', function (e) {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
  }, false)
  btn = createBtn()
  btn.innerHTML = 'Recognize'
  btn.addEventListener('click', function (e) {
    recognize()
  }, false)
  container.appendChild(btnDiv)

  // h2 header
  par = document.createElement('h2')
  par.innerHTML = 'How does this work?'
  par.style.fontWeight = '400'
  par.style.fontSize = '1.4em'
  container.appendChild(par)

  par = document.createElement('p')
  par.innerHTML = [
    'This demo does handwritten digit recognition by evaluating a Convolutional Neural Network on the GPU with WebGL.',
    'The network was trained in TensorFlow <a href="google.com">by this script</a>, and the network was then reimplemented on the GPU by hand with WebGL.',
    'The main purpose of the demo was to demonstate how our WebGL framework',
    '<a href="https://github.com/mikolalysenko/regl">regl</a> can be used to greatly simplify GPGPU programming in WebGL.',
    'The secondary purpose was to test whether evaluating Deep Learning networks in WebGL is doable.',
    'To our knowledge, our implementation is the first implementation ever to attempt GPU accelerating neural networks with WebGL',
    'And we hope that this implementation will provide a foundation for people who, like us, wish to experiment with Deep Learning and WebGL',
    'The GPU implementation can be found <a href="google.com">here</a>'
  ].join('\n')
  container.appendChild(par)

  // add canvas listeners.
  canvas.addEventListener('mousemove', function (e) {
    canvasListener('move', e)
  }, false)
  canvas.addEventListener('mousedown', function (e) {
    canvasListener('down', e)
  }, false)
  canvas.addEventListener('touchstart', function (e) {
    canvasListener('down', e)
  }, false)
  canvas.addEventListener('touchend', function (e) {
    canvasListener('up', e)
  }, false)
  canvas.addEventListener('touchmove', function (e) {
    canvasListener('move', e)
  }, false)
  canvas.addEventListener('mouseup', function (e) {
    canvasListener('up', e)
  }, false)
  canvas.addEventListener('mouseout', function (e) {
    canvasListener('out', e)
  }, false)

  var lineWidth = 20
  var isMousedown = false

  function drawCircle (e) {
    var x = e.pageX - canvas.offsetLeft
    var y = e.pageY - canvas.offsetTop

    ctx.beginPath()
    ctx.lineWidth = 1
    ctx.arc(x, y, lineWidth / 2, 0, 2 * Math.PI)
    ctx.stroke()
    ctx.closePath()
    ctx.fill()
  }

  function canvasListener (cmd, e) {
    if (cmd === 'down') {
      isMousedown = true
      drawCircle(e)
    } else if (cmd === 'up') {
      isMousedown = false
    } else if (cmd === 'move' && isMousedown) {
      drawCircle(e)
      e.preventDefault() // prevent any scrolling.
    }
  }

  // computes center of mass of digit, for centering
  // note 1 stands for black (0 white) so we have to invert.
  function centerImage (img) {
    var meanX = 0
    var meanY = 0
    var rows = img.length
    var columns = img[0].length
    var sumPixels = 0

    for (var y = 0; y < rows; y++) {
      for (var x = 0; x < columns; x++) {
        var pixel = (1 - img[y][x])
        sumPixels += pixel

        meanY += y * pixel
        meanX += x * pixel
      }
    }

    meanX /= sumPixels
    meanY /= sumPixels

    var dY = Math.round(rows / 2 - meanY)
    var dX = Math.round(columns / 2 - meanX)
    return {transX: dX, transY: dY}
  }

  // given grayscale image, find bounding rectangle of digit defined
  // by above-threshold surrounding
  function getBoundingRectangle (img, threshold) {
    var rows = img.length
    var columns = img[0].length
    var minX = columns
    var minY = rows
    var maxX = -1
    var maxY = -1
    for (var y = 0; y < rows; y++) {
      for (var x = 0; x < columns; x++) {
        if (img[y][x] < threshold) {
          if (minX > x) minX = x
          if (maxX < x) maxX = x
          if (minY > y) minY = y
          if (maxY < y) maxY = y
        }
      }
    }
    return {minY: minY, minX: minX, maxY: maxY, maxX: maxX}
  }

  // take canvas image and convert to grayscale. Mainly because my
  // own functions operate easier on grayscale, but some stuff like
  // resizing and translating is better done with the canvas functions
  function imageDataToGrayscale (imgData) {
    var grayscaleImg = []
    for (var y = 0; y < imgData.height; y++) {
      grayscaleImg[y] = []
      for (var x = 0; x < imgData.width; x++) {
        var offset = y * 4 * imgData.width + 4 * x
        var alpha = imgData.data[offset + 3]
        // weird: when painting with stroke, alpha == 0 means white
        // alpha > 0 is a grayscale value in that case I simply take the R value
        if (alpha === 0) {
          imgData.data[offset + 0] = 255
          imgData.data[offset + 1] = 255
          imgData.data[offset + 2] = 255
        }
        imgData.data[offset + 3] = 255
        // simply take red channel value. Not correct, but works for
        // black or white images.
        grayscaleImg[y][x] = imgData.data[y * 4 * imgData.width + x * 4 + 0] / 255
      }
    }
    return grayscaleImg
  }

  function recognize () {
    // NOTE: We need to correctly center and downscale the image before we can feed it to the network.
    // The below code for doing that is not my code, but is MIT-licensed code adapted from this demo:
    // http://myselph.de/neuralNet.html

    // convert RGBA image to a grayscale array, then compute bounding rectangle and center of mass
    var imgData = ctx.getImageData(0, 0, 280, 280)
    var grayscaleImg = imageDataToGrayscale(imgData)

    var boundingRectangle = getBoundingRectangle(grayscaleImg, 0.01)
    //  console.log('boundingRectangle ', boundingRectangle)

    var trans = centerImage(grayscaleImg) // [dX, dY] to center of mass
    //  console.log('trans ', trans)

    // copy image to hidden canvas, translate to center-of-mass, then
    // scale to fit into a 200x200 box (see MNIST calibration notes on
    // Yann LeCun's website)
    var canvasCopy = document.createElement('canvas')
    canvasCopy.width = imgData.width
    canvasCopy.height = imgData.height
    var copyCtx = canvasCopy.getContext('2d')
    var brW = boundingRectangle.maxX + 1 - boundingRectangle.minX
    var brH = boundingRectangle.maxY + 1 - boundingRectangle.minY
    var scaling = 190 / (brW > brH ? brW : brH)
    // scale
    copyCtx.translate(canvas.width / 2, canvas.height / 2)
    copyCtx.scale(scaling, scaling)
    copyCtx.translate(-canvas.width / 2, -canvas.height / 2)
    // translate to center of mass
    copyCtx.translate(trans.transX, trans.transY)

    // default take image from original canvas
    copyCtx.drawImage(ctx.canvas, 0, 0)

    // now bin image into 10x10 blocks (giving a 28x28 image)
    imgData = copyCtx.getImageData(0, 0, 280, 280)
    grayscaleImg = imageDataToGrayscale(imgData)

    var nnInput = new Array(784)
    for (var y = 0; y < 28; y++) {
      for (var x = 0; x < 28; x++) {
        var mean = 0
        for (var v = 0; v < 10; v++) {
          for (var h = 0; h < 10; h++) {
            mean += grayscaleImg[y * 10 + v][x * 10 + h]
          }
        }

        mean = (1 - mean / 100) // average and invert
        nnInput[x + y * 28] = mean
      }
    }

    // after we have processed the canvas, we can actually run the network now:
    var res = cnnGpu(nnInput)
    console.log('result: ', res)

    // output result to HTML
    var actual = res.indexOf(Math.max.apply(null, res))
    digitPar.innerHTML = actual + ''
  }
})
