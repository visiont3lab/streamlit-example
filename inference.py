import onnxruntime as rt
import numpy as np

class modelInference:

  def __init__(self,path2model):
    self.sess = rt.InferenceSession(path2model)
    self.input_name = self.sess.get_inputs()[0].name
    self.input_shape = self.sess.get_inputs()[0].shape
    self.input_type = self.sess.get_inputs()[0].type
    self.output_name = self.sess.get_outputs()[0].name
    self.output_shape = self.sess.get_outputs()[0].shape
    self.output_type = self.sess.get_outputs()[0].type

  def print_model_info(self):
    # Input informations
    print("input name", self.input_name)
    print("input shape", self.input_shape)
    print("input type", self.input_type)
    # Outpout informations
    print("output name", self.output_name)
    print("output shape", self.output_shape)
    print("output type", self.output_type)

  def predict(self, x):
    x = x.astype(np.float32)
    res = self.sess.run([self.output_name], {self.input_name: x})
    return res


if __name__ == "__main__":
  path2model = "models/manuel.onnx"
  mi = modelInference(path2model)
  mi.print_model_info()
  x = np.random.random((1,3,224,224))
  res = mi.predict(x)
  print(res)
