var zeros = require('zeros')

module.exports.cpuSoftmax = function (x) {
  var len = x.shape[1]
  var i

  var sum = 0 // the denominator.
  for (i = 0; i < len; i++) {
    sum += Math.exp(x.get(0, i))
  }

  var res = zeros([1, len])

  for (i = 0; i < len; i++) {
    res.set(0, i, Math.exp(x.get(0, i)) / sum)
  }

  return res
}

function float16ToFloat (h) {
  var s = (h & 0x8000) >> 15
  var e = (h & 0x7C00) >> 10
  var f = h & 0x03FF

  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10))
  } else if (e === 0x1F) {
    return f ? NaN : ((s ? -1 : 1) * Infinity)
  }

  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + (f / Math.pow(2, 10)))
}

// convert sequence of 16-bit floats to 32-bit floats.
function binaryParse (data) {
  var length = new Int32Array(data, 0, 1)[0]
  var ints = new Int16Array(data.slice(4, 4 + length * 2))

  var parsed = []
  for (var i = 0; i < length; i++) {
    parsed.push(float16ToFloat(ints[i]))
  }

  return parsed
}

module.exports.loadData = function (cb) {
  require('resl')({
    manifest: {
      wConv1Data: {
        type: 'binary',
        src: './w_conv1.bin',
        parser: binaryParse
      },
      bConv1Data: {
        type: 'binary',
        src: './b_conv1.bin',
        parser: binaryParse
      },
      wFc1Data: {
        type: 'binary',
        src: './w_fc1.bin',
        parser: binaryParse
      },
      bFc1Data: {
        type: 'binary',
        src: './b_fc1.bin',
        parser: binaryParse
      },
      wFc2Data: {
        type: 'binary',
        src: './w_fc2.bin',
        parser: binaryParse
      },
      bFc2Data: {
        type: 'binary',
        src: './b_fc2.bin',
        parser: binaryParse
      }
    },
    onDone: function (jsonData) {
      cb(jsonData)
    }
  })
}
