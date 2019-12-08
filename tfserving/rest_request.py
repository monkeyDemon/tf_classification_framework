from __future__ import print_function

import base64
import requests

# The server URL specifies the endpoint of your server running the model
# TODO:
SERVER_URL = 'http://modify your ip:port/v1/models/modelname:predict'  

# The image URL is the location of the image we should send to the server for test
IMAGE_URL = 'path of a image'


def main():

    image_sign = 'local'
    if image_sign == 'download':
        # Download the image
        dl_request = requests.get(IMAGE_URL, stream=True)
        dl_request.raise_for_status()
        image_str = dl_request.content
    else:
        # read image from local
        with open(IMAGE_URL, 'rb') as f:
            image_str = f.read()
        
  
    # Compose a JSON Predict request (send JPEG image in base64).
    jpeg_bytes = base64.b64encode(image_str).decode('utf-8')
    predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
  
    # Send few actual requests and report average latency.
    total_time = 0
    num_requests = 10
    for _ in range(num_requests):
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()
        prediction = response.json()['predictions'][0]
  
    print('Prediction class: {}, avg latency: {} ms'.format(
        prediction['scores'], (total_time*1000)/num_requests))
  

if __name__ == '__main__':
  main()
