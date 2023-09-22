import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="app\\src\main\\assets\\lite-model_mobilenetv2-dm05-coco_dr_1.tflite")
interpreter.allocate_tensors()

# Print input shape and type
inputs = interpreter.get_input_details()
print('{} input(s):'.format(len(inputs)))
for i in range(0, len(inputs)):
    print('{} {}'.format(inputs[i]['shape'], inputs[i]['dtype']))

# Print output shape and type
outputs = interpreter.get_output_details()
print('\n{} output(s):'.format(len(outputs)))
for i in range(0, len(outputs)):
    print('{} {}'.format(outputs[i]['shape'], outputs[i]['dtype']))


# deeplabv3
# https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1
# lite-model_deeplabv3_1_metadata_2.tflite
# 1 input(s):
# [  1 257 257   3] <class 'numpy.float32'>

# 1 output(s):
# [  1 257 257  21] <class 'numpy.float32'>


### deeplabv3-xception65
# https://tfhub.dev/sayakpaul/lite-model/deeplabv3-xception65/1/default/2
# lite-model_deeplabv3-xception65_1_default_2.tflite
    # 1 input(s):
    # [  1 513 513   3] <class 'numpy.float32'>

    # 1 output(s):
    # [  1 129 129  21] <class 'numpy.float32'>


### deeplabv3-mobilenetv2_dm05
# https://tfhub.dev/sayakpaul/lite-model/deeplabv3-mobilenetv2_dm05/1/default/2
# lite-model_deeplabv3-mobilenetv2_dm05_1_default_2.tflite
# 1 input(s):
# [  1 513 513   3] <class 'numpy.float32'>

# 1 output(s):
# [  1 513 513  21] <class 'numpy.float32'>


### deeplabv3-mobilenetv2
# https://tfhub.dev/sayakpaul/lite-model/deeplabv3-mobilenetv2/1/default/1
# lite-model_deeplabv3-mobilenetv2_1_default_1.tflite
# 1 input(s):
# [  1 513 513   3] <class 'numpy.float32'>

# 1 output(s):
# [ 1 65 65 21] <class 'numpy.float32'>


### mobilenetv2-dm05-coco
# https://tfhub.dev/sayakpaul/lite-model/mobilenetv2-dm05-coco/dr/1
# lite-model_mobilenetv2-dm05-coco_dr_1.tflite
# 1 input(s):
# [  1 513 513   3] <class 'numpy.float32'>

# 1 output(s):
# [  1 513 513  21] <class 'numpy.float32'>