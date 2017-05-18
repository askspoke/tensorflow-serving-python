
from tensorflow_serving_python.client import TFClient

tfserving_client = TFClient('127.0.0.1', 9000)

img_data = open('/Users/mihail/data/solids/6e77783879364a76786d31465f346462.png','rb').read()
pred = tfserving_client.make_prediction(img_data, timeout=60, name='segmentation', signature_name='predict_images')
print(pred['scores'].shape)