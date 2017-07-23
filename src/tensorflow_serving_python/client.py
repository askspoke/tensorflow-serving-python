#import tensorflow as tf
from .tensor_utils.tensor_utils import make_tensor_proto, make_nd_array
from grpc.beta import implementations

from tensorflow_serving_python.protos import prediction_service_pb2, predict_pb2
from tensorflow_serving_python.proto_util import copy_message

class TFClient(object):
    """
    TFClient class to use for RPC calls
    """
    def __init__(self, host, port):
        """
        Setup stuff
        :param host: Server hostname
        :param port: Server port
        :return:
        """
        self.host = host
        self.port = port

        # Setup channel
        self.channel = implementations.insecure_channel(self.host, int(self.port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)

    def execute(self, request, timeout=10.0):
        """
        Execture the RPC request
        :param request: Request proto
        :param timeout: Timeout in seconds to wait for more batches to pile up
        :return: Prediction result
        """
        return self.stub.Predict(request, timeout)

    def make_prediction(self, data, name='inception', timeout=10., convert_to_dict=True, signature_name=None):
        """
        Make a prediction on a buffer full of image data (tested .jpg as of now)
        :param data: Data buffer
        :param name: Name of the model_spec to use
        :param timeout: Timeout in seconds to wait for more batches to pile up
        :return: Prediction result
        """
        request = predict_pb2.PredictRequest()
        request.model_spec.name = name
        if signature_name:
            request.model_spec.signature_name = signature_name
        proto = make_tensor_proto(data, shape=[1])

        # TODO dst.CopyFrom(src) fails here because we compile custom protocolbuffers
        # TODO Proper compiling would speed up the next line by a factor of 10
        copy_message(proto, request.inputs['image_bytes'])
        response = self.execute(request, timeout=timeout)

        if not convert_to_dict:
            return response

        # Convert to friendly python object
        results_dict = {}
        for key in response.outputs:
            tensor_proto = response.outputs[key]
            nd_array = make_nd_array(tensor_proto)
            results_dict[key] = nd_array

        return results_dict
