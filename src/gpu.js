var ndarray = require('ndarray')
var cpuSoftmax = require('./common.js').cpuSoftmax

module.exports = function (regl, d) {
  var WIDTH = 28
  var HEIGHT = 28

  /*
    We use RGBA32f textures throughout the computations.
    And we use RGBA, because
    you can't render to a single-channel texture in WebGL.
  */
  function createTexture (width, height) {
    return regl.texture({
      width: width,
      height: height,
      format: 'rgba',
      type: 'float',
      mag: 'nearest',
      min: 'nearest',
      wrap: 'clamp'
    })
  }

  function getFbo (fbo) {
    var a = []
    regl({framebuffer: fbo})(function () {
      var arr = regl.read()
      for (var i = 0; i < arr.length; i += 4) {
        a.push(arr[i])
      }
    })
    return ndarray(a, [fbo.color[0].height, fbo.color[0].width])
  }

  // get all the pixel values of the red channel in the texture.
  function getTex (tex) {
    return getFbo(regl.framebuffer({color: tex}))
  }

  // this is the basis of all the GPGPU commands that we write.
  var baseGpu = {
    vert: [
      'precision mediump float;',
      'attribute vec2 position;',
      'varying vec2 vUv;',
      'void main () {',
      '  vUv = 0.5 * (position + 1.0);',
      '  gl_Position = vec4(position, 0, 1);',
      '}'
    ].join('\n'),

    attributes: {
      // we render a full-screen triangle.
      position: [ -4, -4, 4, -4, 0, 4 ]
    },
    count: 3,

    // and we always output to an FBO
    framebuffer: regl.prop('outFbo')
  }

  // gpu matrix multiplication. Returns a*b
  function gpuMul (a, b) {
    if (a.width !== b.height) {
      throw Error('Impossible matrix mul!')
    }
    var outTex = createTexture(b.width, a.height)

    var obj = baseGpu

    obj.frag = [
      'precision mediump float;',

      'uniform sampler2D aTex;',
      'uniform sampler2D bTex;',

      'varying vec2 vUv;',

      '#define oDim vec2(' + b.width + ',' + a.height + ')',

      '#define aDim vec2(' + a.width + ',' + a.height + ')',

      '#define bDim vec2(' + b.width + ',' + b.height + ')',

      '#define aDimi ivec2(' + a.width + ',' + a.height + ')',

      'void main () {',
      // set uv to top-left corner of texel.
      '  vec2 uv = vUv;',

      '  float x = uv.x * oDim.x;',
      '  float y = uv.y * oDim.y;',

      '  float sum = 0.0;',

      '  for (int i = 0; i < aDimi.x; i++) {',
           // multiply row i of a with col i of b.
      '    float a = texture2D(aTex, vec2(float(i) / aDim.x, y / aDim.y)).x;',
      '    float b = texture2D(bTex, vec2(x / bDim.x, float(i) / bDim.y)).x;',

      '    sum += a*b;',
      '  }',

      'gl_FragColor = vec4(sum);',
      '}'].join('\n')

    obj.uniforms = {
      aTex: regl.prop('aTex'),
      bTex: regl.prop('bTex')
    }

    var cmd = regl(obj)

    return {
      out: outTex,
      func: function () {
        var outFbo = regl.framebuffer({color: outTex})
        cmd({aTex: a, bTex: b, outFbo: outFbo})
      }
    }
  }

  // gpu component-wise matrix add.
  function gpuAdd (a, b) {
    if (a.width !== b.width || a.height !== b.height) {
      throw Error('Impossible matrix add!')
    }
    var outTex = createTexture(a.width, a.height)

    var obj = baseGpu

    obj.frag = [
      'precision mediump float;',

      'uniform sampler2D aTex;',
      'uniform sampler2D bTex;',

      'varying vec2 vUv;',

      'void main () {',
      '  float a = texture2D(aTex, vUv).x;',
      '  float b = texture2D(bTex, vUv).x;',

      '  float res = a + b;',

      'gl_FragColor = vec4(res);',
      '}'
    ].join('\n')

    obj.uniforms = {
      aTex: regl.prop('aTex'),
      bTex: regl.prop('bTex')
    }

    var cmd = regl(obj)

    return {
      out: outTex,
      func: function () {
        var outFbo = regl.framebuffer({color: outTex})
        cmd({aTex: a, bTex: b, outFbo: outFbo})
      }
    }
  }

  // do relu on every element in a.
  // where relu = max(0,x)
  function gpuRelu (a) {
    var outTex = createTexture(a.width, a.height)

    var obj = baseGpu

    obj.frag = [
      'precision mediump float;',

      'uniform sampler2D aTex;',

      'varying vec2 vUv;',

      'void main () {',
      '  float a = texture2D(aTex, vUv).x;',
      '  float res = max(0.0, a);',
      '  gl_FragColor = vec4(res);',
      '}'
    ].join('\n')

    obj.uniforms = {
      aTex: regl.prop('aTex')
    }

    var cmd = regl(obj)

    return {
      out: outTex,
      func: function () {
        var outFbo = regl.framebuffer({color: outTex})
        cmd({aTex: a, outFbo: outFbo})

        return outTex
      }
    }
  }

  // flatten a tensor into a column vector,
  // and put that column vector in a texture.
  function gpuFlatten (W) {
    // output texture dimensions.
    var oWidth = W.length * W[0].width * W[0].height
    var oHeight = 1
    var outTex = createTexture(oWidth, oHeight)

    var obj = baseGpu

    obj.frag = [
      'precision mediump float;',

      'uniform sampler2D wTex;',
      'varying vec2 vUv;',

      'uniform float i;', // which layer we are processing.

      '#define oDim vec2(' + oWidth + ',' + oHeight + ')',

      '#define wDim vec2(' + W[0].width + ',' + W[0].height + ')',

      '#define wRcp vec2(1.0 / float(' + W[0].width + '), 1.0 / float(' + W[0].height + '))',

      '#define wSize float(' + W[0].width + '*' + W[0].height + ')',

      'void main () {',
      '  vec2 uv = vUv;',

      '  float elem = uv.x * oDim.x;',

      // are we processing the right layer?
      '  if(((floor(elem / wSize)) - i) < 0.001) {',
      '    vec4 res = vec4(1.0);',
      '    elem = mod(elem, wSize);',

      // find the value in the original texture, that is corresponding
      // to the value in the flattened texture.
      '    float x = (mod(elem, wDim.x)) / wDim.x;',
      '    float y = (elem / wDim.x) / wDim.y;',
      '    res = vec4(texture2D(wTex, vec2(x,y) ).x);',

      '    gl_FragColor = vec4(res);',
      '  } else {',
      // wrong layer, so do nothing!
      '    discard;',
      '  }',
      '}'
    ].join('\n')

    obj.uniforms = {
      i: regl.prop('i'),
      wTex: regl.prop('tex')
    }

    var pass = regl(obj)
    return {
      out: outTex,
      func: function () {
        var outFbo = regl.framebuffer({color: outTex})
        for (var i = 0; i < W.length; i++) {
          // NOTE: To simplify the implementation, we use 'discard'.
          // But that also results in a bit of overdraw.
          // But by rewriting the algorithm so that it does not use
          // 'discard', some performance can be gained.
          // But for this demo simple demo, that performance gain will
          // be minimal, so we do this simpler implementation.
          pass({tex: W[i], outFbo: outFbo, i: i})
        }
      }
    }
  }

  // implements max-pool layer.
  // with stride 2x2, and patch size 2x2.
  function gpuMaxPool (W) {
    var outLayers = []
    for (var i = 0; i < W.length; i++) {
      var outTex = createTexture(WIDTH / 2, HEIGHT / 2)
      outLayers.push(outTex)
    }

    var obj = baseGpu

    obj.frag = [
      'precision mediump float;',

      'uniform sampler2D wTex;',
      'varying vec2 vUv;',

      '#define wDim vec2(' + W[0].width + ',' + W[0].height + ')',

      'void main () {',
      '  vec2 wRcp = vec2(1.0 / wDim.x, 1.0 / wDim.y);',
      '  vec2 uv = vUv;',

      '  float a = texture2D(wTex, uv + wRcp * vec2(0.0, 0.0)).x;',
      '  float b = texture2D(wTex, uv + wRcp * vec2(1.0, 0.0)).x;',
      '  float c = texture2D(wTex, uv + wRcp * vec2(0.0, 1.0)).x;',
      '  float d = texture2D(wTex, uv + wRcp * vec2(1.0, 1.0)).x;',

      '  float res = max(max(a, b), max(c, d));',
      '  gl_FragColor = vec4(res);',

      '}'
    ].join('\n')

    obj.uniforms = {
      wTex: regl.prop('tex')
    }

    var cmd = regl(obj)

    return {
      out: outLayers,
      func: function () {
        var outFbo = regl.framebuffer({})

        for (var i = 0; i < W.length; i++) {
          outFbo({color: outLayers[i]})
          // do a max-pool operations for all the layers.
          cmd({tex: W[i], outFbo: outFbo})
        }
      }
    }
  }

  // Do conv2d, bias and relu in a single pass,  for increased performance.
  // CONV patch size is 2x2, and the stride is 1x1.
  function gpuConv2dBiasRelu (W, bias) {
    var outLayers = []
    for (var i = 0; i < W.length; i++) {
      var outTex = createTexture(WIDTH, HEIGHT)
      outLayers.push(outTex)
    }

    var obj = baseGpu
    var kernelRad = Math.floor(W[0].width / 2.0)

    obj.frag = [
      'precision mediump float;',

      'uniform sampler2D tex;',
      'uniform sampler2D kernel;',
      'uniform sampler2D bias;',

      'varying vec2 vUv;',

      'uniform float i;', // current layer index
      '#define N float(' + W.length + ')',

      '#define kRad int(' + kernelRad + ')',
      '#define kDim vec2(' + W[0].width + ',' + W[0].height + ')',

      '#define tDim vec2(' + WIDTH + ',' + HEIGHT + ')',

      'void main () {',
      '  vec2 tRcp = vec2(1.0 / tDim.x, 1.0 / tDim.y);',
      '  vec2 kRcp = vec2(1.0 / kDim.x, 1.0 / kDim.y);',
      '  vec2 uv = vUv;',

      '  float sum = 0.0;',

      '  for( int ir = -kRad; ir <= +kRad; ir++) {',
      '    for( int ic = -kRad; ic <= +kRad; ic++) {',
      '      vec2 tUv = uv + vec2(float(ic), float(ir)) * tRcp;',
      '      float a = texture2D(tex, tUv).x;',

      '      vec2 kUv = (vec2(float(ic), float(ir)) + vec2(float(kRad))) * kRcp;',
      '      kUv += kRcp * 0.5;', // make sure we sample from the texel center!
      '      float b = texture2D(kernel, kUv).x;',

      '      float e = 0.01;',
      // we use zero padding!
      '      if(tUv.x < (0.0-e) || tUv.y < (0.0-e) || tUv.x > (1.0+e) || tUv.y > (1.0+e)) {',
      '        a = 0.0;',
      '      }',

      '      sum += a * b;',
      '    }',
      '  }',

      // add bias
      '  sum += texture2D(bias, vec2(i / N, 0.0)).x;',

      // relu
      '  sum = max(sum, 0.0);',

      '  gl_FragColor = vec4(sum);',
      '}'
    ].join('\n')

    obj.uniforms = {
      tex: regl.prop('tex'),
      kernel: regl.prop('kernel'),
      bias: regl.prop('bias'),
      i: regl.prop('i')
    }

    var pass = regl(obj)

    return {
      out: outLayers,
      func: function (x) {
        var outFbo = regl.framebuffer({})

        for (var i = 0; i < W.length; i++) {
          outFbo({color: outLayers[i]})
          pass({tex: x, kernel: W[i], bias: bias, outFbo: outFbo, i: i})
        }
        return outLayers
      }
    }
  }

  function createTensor (W, shape) {
    var i

    var textureData = []
    var g
    if (shape.length === 3) {
      var textureSize = shape[1] * shape[2]
      var layers = []

      for (var l = 0; l < shape[0]; l++) {
        // go through all the layers,
        // and create a texture for each layer.
        textureData = []

        for (i = 0; i < textureSize; i++) {
          g = W[textureSize * l + i]
          textureData.push(g, g, g, g)
        }

        var layer = regl.texture({
          width: shape[1],
          height: shape[2],
          data: textureData,
          format: 'rgba',
          type: 'float',
          mag: 'nearest',
          min: 'nearest'
        })
        layers.push(layer)
      }
      return layers
    } else {
      // create 2D tensor. So a single texture.
      textureData = []
      for (i = 0; i < W.length; i++) {
        g = W[i]
        textureData.push(g, g, g, g)
      }

      var tex = regl.texture({
        width: shape[1],
        height: shape[0],
        data: textureData,
        format: 'rgba',
        type: 'float',
        mag: 'nearest',
        min: 'nearest'
      })

      return tex
    }
  }

  //
  //  In advance, we create and cache all the tensors that we need
  //
  var wConv1 = createTensor(d.wConv1Data, [16, 5, 5])
  var bConv1 = createTensor(d.bConv1Data, [1, 16])
  var wFc1 = createTensor(d.wFc1Data, [14 * 14 * 16, 64])
  var wFc2 = createTensor(d.wFc2Data, [64, 10])
  var bFc2 = createTensor(d.bFc2Data, [1, 10])
  var bFc1 = createTensor(d.bFc1Data, [1, 64])

  //
  // We also create all draw commands.
  //
  var gpuConv2dBiasReluFunc = gpuConv2dBiasRelu(wConv1, bConv1)
  var gpuMaxPoolFunc = gpuMaxPool(gpuConv2dBiasReluFunc.out)
  var gpuFlattenFunc = gpuFlatten(gpuMaxPoolFunc.out)

  var gpuMulFunc1 = gpuMul(gpuFlattenFunc.out, wFc1)
  var gpuAddFunc1 = gpuAdd(gpuMulFunc1.out, bFc1)

  var gpuReluFunc = gpuRelu(gpuAddFunc1.out)

  var gpuMulFunc2 = gpuMul(gpuReluFunc.out, wFc2)
  var gpuAddFunc2 = gpuAdd(gpuMulFunc2.out, bFc2)

  // warm up the regl drawing commands, by running the entire network once,
  // for an array of zeros.
  cnnGpu(Array.apply(null, Array(WIDTH * HEIGHT)).map(Number.prototype.valueOf, 0))

  function cnnGpu (xSrc) {
    var x = createTensor(xSrc, [28, 28])

    // since we have already created the resources and draw commands,
    // running CNN will be fast.
    gpuConv2dBiasReluFunc.func(x)

    gpuMaxPoolFunc.func()

    gpuFlattenFunc.func()

    gpuMulFunc1.func()

    gpuAddFunc1.func()

    gpuReluFunc.func()

    gpuMulFunc2.func()

    gpuAddFunc2.func()

    // we will only be doing softmax on 10 elements, so very little will be
    // gained from GPU accelerating softmax. So let us simplify things
    // and do it on the CPU:
    var res = getTex(gpuAddFunc2.out)
    res = cpuSoftmax(res)

    return res.data
  }

  return cnnGpu
}
