var loadData = require('./common.js').loadData
var canvas = document.body.appendChild(document.createElement('canvas'))
canvas.width = 1
canvas.height = 1

const regl = require('regl')({canvas: canvas, extensions: ['oes_texture_float']})

/*
  This script runs some test cases that verify the implementation of the digit
  recognizer on both the CPU and the GPU.

  The test data is just digits taken from the MNIST dataset.
 */
var testData = [
  [require('./d0.json'), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
  [require('./d1.json'), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
  [require('./d2.json'), [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
  [require('./d3.json'), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
  [require('./d4.json'), [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
]

var container = document.createElement('div')
document.body.appendChild(container)

loadData(function (data) {
  container.innerHTML += ('EVALUATE CNN ON CPU <br>')
  var cnnCpu = require('./cpu.js')(data)

  testData.forEach(function (d, i) {
    var res = cnnCpu(d[0])
    var arr = Array.prototype.slice.call(res)
    var actual = arr.indexOf(Math.max.apply(null, arr))

    arr = d[1]
    var expected = arr.indexOf(Math.max.apply(null, arr))
    if (actual === expected) {
      container.innerHTML += ('#' + i + ' PASS <br>')
    } else {
      container.innerHTML += ('#' + i + ' FAIL <br>')
    }
    container.innerHTML += ('result: ' + res + '<br><br>')
  })
  container.innerHTML +=
  ('-----------------------------------------------------------------------------------------------<br><br><br>')

  var cnnGpu = require('./gpu.js')(regl, data)
  container.innerHTML += ('EVALUATE CNN ON GPU <br>')

  testData.forEach(function (d, i) {
    var res = cnnGpu(d[0])

    var arr = Array.prototype.slice.call(res)
    var actual = arr.indexOf(Math.max.apply(null, arr))

    arr = d[1]
    var expected = arr.indexOf(Math.max.apply(null, arr))
    if (actual === expected) {
      container.innerHTML += ('#' + i + ' PASS <br>')
    } else {
      container.innerHTML += ('#' + i + ' FAIL <br>')
    }
    container.innerHTML += ('result: ' + res + '<br><br>')
  })
})
