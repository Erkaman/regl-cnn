var ndarray = require('ndarray')
var cpuSoftmax = require('./common.js').cpuSoftmax
var zeros = require('zeros')
var gemm = require('ndarray-gemm')
var ops = require('ndarray-ops')
var cwise = require('cwise')

/*
  Here, we implement the CNN on the CPU with ndarray

  This was implemented so that we could easily verify the GPU-based
  CNN implementation.
 */
module.exports = function (d) {
  // matrix multiplication of a*b
  function cpuMul (a, b) {
    var res = zeros([a.shape[0], b.shape[1]])
    gemm(res, a, b)
    return res
  }

  function cpuAdd (a, b) {
    var res
    if (a.shape.length === 2) { // if ndarrays with two dimensions.
      // component-wise add.
      res = zeros([a.shape[0], a.shape[1]])
      ops.add(res, a, b)
      return res
    } else { // we consider a to be a bunch of layers.
      res = zeros(a.shape)

      // b[l] is added to all the values in layer l of a.
      for (var l = 0; l < a.shape[0]; l++) {
        var bias = b.get(l) // add this bias to all weights in this layer.

        for (var i = 0; i < a.shape[1]; i++) {
          for (var j = 0; j < a.shape[2]; j++) {
            res.set(l, i, j, a.get(l, i, j) + bias)
          }
        }
      }
      return res
    }
  }

  // implements a very small subset of the conv2d operation from tensorflow
  // stride is always 1x1. And we have zero padding.
  // W is filter
  // x is input image.
  function cpuConv2d (W, x) {
    var outArray = []

    for (var f = 0; f < W.shape[0]; f++) {
      // layer output
      var layer = zeros([x.shape[0], x.shape[1]])
      // layer filter
      var filter = W.pick(f, null, null)

      var rad = Math.floor(filter.shape[0] / 2) // kernel radius

      // apply filter to image(x)
      for (var row = 0; row < x.shape[0]; row++) {
        for (var col = 0; col < x.shape[1]; col++) {
          var sum = 0

          for (var ir = -rad; ir <= +rad; ir++) {
            for (var ic = -rad; ic <= +rad; ic++) {
              var a // from filter
              var b // from original image

              a = filter.get(ir + rad, ic + rad)
              var pr = row + ir
              var pc = col + ic
              if (pr < 0 || pc < 0 || pr >= x.shape[0] || pc >= x.shape[1]) {
                // we always have zero padding.
                b = 0
              } else {
                b = x.get(pr, pc)
              }
              sum += a * b
            }
          }

          layer.set(row, col, sum)
        }
      }
      outArray = outArray.concat(Array.prototype.slice.call(layer.data))
    }

    var data = ndarray(outArray, [W.shape[0], x.shape[0], x.shape[1]])

    return data
  }

  // 2x2 max-pool with stride 2x2.
  // W will be a bunch of layers, and we apply the
  // max-pool operation to each one of then,
  // separately
  function cpuMaxPool (W) {
    var outArray = []

    var outWidth = W.shape[1] / 2
    var outHeight = W.shape[2] / 2

    for (var f = 0; f < W.shape[0]; f++) { // go through all layers.
      var layer = zeros([outWidth, outHeight])

      for (var row = 0; row < outHeight; row++) {
        for (var col = 0; col < outWidth; col++) {
          // stride 2x2
          var r = 2 * row
          var c = 2 * col

          var x = W.get(f, r + 0, c + 0)
          var y = W.get(f, r + 1, c + 0)
          var z = W.get(f, r + 0, c + 1)
          var w = W.get(f, r + 1, c + 1)

          var m = Math.max(Math.max(x, y), Math.max(z, w))

          layer.set(row, col, m)
        }
      }
      outArray = outArray.concat(Array.prototype.slice.call(layer.data))
    }

    var data = ndarray(outArray, [W.shape[0], outWidth, outHeight])

    return data
  }

  // relu on all the elements in argument.
  // where relu is defined as max(x,0)
  var cpuRelu = cwise({
    args: ['array'],
    body: function (a) {
      a = Math.max(0, a)
    }
  })

  function cnnCpu (xSrc) {
    var x = ndarray(xSrc, [28, 28])

    var wConv1 = ndarray(d.wConv1Data, [16, 5, 5])
    var bConv1 = ndarray(d.bConv1Data, [16])

    // conv, add bias, relu
    var res = cpuConv2d(wConv1, x)
    res = cpuAdd(res, bConv1)
    cpuRelu(res)

    // max pool
    res = cpuMaxPool(res)

    // densely connected layer.
    var wFc1 = ndarray(d.wFc1Data, [14 * 14 * 16, 64])
    var bFc1 = ndarray(d.bFc1Data, [1, 64])
    res = ndarray(res.data, [1, 14 * 14 * 16])
    res = cpuMul(res, wFc1)
    res = cpuAdd(res, bFc1)
    cpuRelu(res)

    // readout layer
    var wFc2 = ndarray(d.wFc2Data, [64, 10])
    var bFc2 = ndarray(d.bFc2Data, [1, 10])
    res = cpuMul(res, wFc2)
    res = cpuAdd(res, bFc2)

    // finally, softmax.
    res = cpuSoftmax(res)
    return res.data
  }

  return cnnCpu
}
