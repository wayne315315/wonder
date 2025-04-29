from pathlib import Path
import shutil
import itertools
import subprocess
import time
import os
import contextlib

import grpc
import requests
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


# Server config
BASE_DIR = Path(__file__).parent
CONFIG_DIR = Path(BASE_DIR, "config")
SERVE_BASE_DIR = Path(BASE_DIR, "serve")
TARGET_DIR = Path("/", "models")
GRPC_PORT = 8500
REST_API_PORT = 8501
HOST = "localhost"
NAME_STATE = "state" 
NAME_HAND = "hand"

# launch tensorflow serving
def launch_server(grpc_port=GRPC_PORT, rest_api_port=REST_API_PORT):
    # Ensure SERVE_BASE_DIR exists
    SERVE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    # cmd
    cmd = ["docker run -d --rm -p {grpc_port}:{grpc_port} -p {rest_api_port}:{rest_api_port} --gpus all"]
    cmd.append("--name tensorflow_serving")
    cmd.append("--mount type=bind,source={source},target={target}")
    cmd.append("--mount type=bind,source={config_dir},target=/etc/config")
    cmd.append("tensorflow/serving:2.17.1-gpu")
    cmd.append("--port=8500")
    cmd.append("--rest_api_port=8501")
    cmd.append("--grpc_max_threads=65536")
    cmd.append("--num_load_threads=8")
    cmd.append("--num_unload_threads=8")
    cmd.append("--file_system_poll_wait_seconds=1")
    cmd.append("--model_config_file_poll_wait_seconds=1")
    cmd.append("--enable_batching=true")
    cmd.append("--batching_parameters_file=/etc/config/batching_parameters.pbtxt")
    cmd.append("--model_config_file=/etc/config/models_config.pbtxt")
    cmd = " ".join(cmd)
    cmd = cmd.format(
        grpc_port=grpc_port,
        rest_api_port=rest_api_port,
        source=str(SERVE_BASE_DIR),
        config_dir=str(CONFIG_DIR),
        target=str(TARGET_DIR)
    )
    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=True ,
        check=True # Raise an error if the command fails
    )
    container_id = process.stdout.strip()[:12]
    print("Running tensorflow server in container id:", container_id)
    return container_id

def kill_server():
    cmd = "docker stop $(docker ps -qf name=tensorflow_serving)"
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    process.wait()
    if process.returncode == 0:
        print("TF Server stopped successfully")
    else:
        print("Error stopping TF server")

def probe(serve_name, serve_version=None, verbose=False):
    url = "http://localhost:8501/v1/models/{serve_name}".format(serve_name=serve_name)
    healthy = False
    try:
        resp = requests.get(url)
    except Exception as e:
        if verbose:
            print(f"Error during health check: {e}")
    else:
        if resp.status_code == 200:
            if verbose:
                for serve in resp.json()["model_version_status"]:
                    print("verions %s: %s" % (serve["version"], serve["state"]))
            if serve_version is None:
                healthy = any([serve["state"] == "AVAILABLE" for serve in resp.json()["model_version_status"]])
            else:
                healthy = any([(serve["state"] == "AVAILABLE") and (serve["version"] == str(serve_version)) for serve in resp.json()["model_version_status"]])
        else:
            print("Health check failed: ", resp.status_code, resp.json())
    return healthy

# get serving function for endpoints
def get_serving_fn(model, s_dim=None, h_dim=None):
    input_signature = (
        tf.TensorSpec(shape=[None, s_dim, 7], dtype=tf.int32, name=NAME_STATE), # vs
        tf.TensorSpec(shape=[None, h_dim], dtype=tf.int32, name=NAME_HAND) # hs
    )
    serving_fn = tf.function(input_signature=input_signature)(lambda state, hand: model.predict_move(state, hand))
    return serving_fn

def clean_archive(serve_name):
    serve_dir = Path(SERVE_BASE_DIR, serve_name)
    shutil.rmtree(serve_dir)
    print("Delete model archive:", serve_name)

def export_archive(serve_name, model, epoch, kill_old_model=True, last_num_version=2):
    # Create a servable
    export_archive = tf.keras.export.ExportArchive()
    # Track the model
    export_archive.track(model)
    # Register endpoints for each (s_dim, h_dim)
    dims_state = [19 * n + 6 for n in range(3,8)]
    dims_hand = [7, 21] 

    for s_dim, h_dim in itertools.product(dims_state, dims_hand):
        export_archive.add_endpoint(
            "serving_%d_%d" % (s_dim, h_dim),
            get_serving_fn(model, s_dim=s_dim, h_dim=h_dim)
        )
    # Register the default endpoint
    export_archive.add_endpoint(
        "serving_default",
        get_serving_fn(model)
    )
    # export SavedModel to servable path
    serve_dir = Path(SERVE_BASE_DIR, serve_name)
    serve_path = Path(serve_dir, str(epoch))
    serve_path.mkdir(parents=True, exist_ok=True)
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            export_archive.write_out(serve_path)
    print("Exported model version:", epoch)
    # kill old model
    if kill_old_model:
        old_version = epoch - last_num_version
        old_serve_path = Path(serve_dir, str(old_version))
        if old_serve_path.exists():
            shutil.rmtree(old_serve_path)
            print("Delete model version:", old_version)

def get_grpc_channel(host=HOST, grpc_port=GRPC_PORT):
    channel_address = f"{host}:{grpc_port}"
    channel = None
    while channel is None:
        try:
            channel = grpc.insecure_channel(channel_address)
        except Exception as e:
            print(f"Error connecting to gRPC server: {e}")
            print("Retrying in 1 seconds...")
            time.sleep(1)
    return channel

def close_grpc_channel(channel):
    if channel:
        channel.close()

def send_grpc_request(states, hands, serve_name, serve_version=None, host=HOST, grpc_port=GRPC_PORT):
    # create grpc channel
    channel = get_grpc_channel(host=host, grpc_port=grpc_port)
    # signature name
    s_dim = states.shape[1]
    h_dim = hands.shape[1]
    signature_name = "serving_%d_%d" % (s_dim, h_dim)
    # create client stub
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # placeholder for response
    moves = None
    # create grpc request
    try:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = serve_name
        request.model_spec.signature_name = signature_name
        # if request.model_spec.version.value is not set, the server will choose the latest version
        if serve_version is not None:
            request.model_spec.version.value = serve_version
        # convert numpy array to tensor proto
        request.inputs[NAME_STATE].CopyFrom(
            tf.make_tensor_proto(states, shape=states.shape, dtype=states.dtype.name)
        )
        request.inputs[NAME_HAND].CopyFrom(
            tf.make_tensor_proto(hands, shape=hands.shape, dtype=hands.dtype.name)
        )
        result_future = stub.Predict.future(request)
        result = result_future.result()
        output_tensor_proto = result.outputs["output_0"]
        moves = tf.make_ndarray(output_tensor_proto)
    except grpc.RpcError as e:
        print(f"\n--- gRPC Request Failed ---")
        print(f"Status Code: {e.code()}")
        print(f"Details: {e.details()}")
    except Exception as e:
        print(f"\n--- An unexpected client error occurred ---")
        print(f"Error: {e}")
    finally:
        close_grpc_channel(channel)
    return moves

if __name__ == "__main__":
    import numpy as np
    from model import ActorCritic
    model_path = Path(BASE_DIR, "model", "ac.keras")
    model = tf.keras.models.load_model(model_path)
    serve_name = "ac"
    last_num_version = 2
    # base model version 0
    export_archive(serve_name, model, 0, last_num_version=last_num_version)
    # launch server
    container_id = launch_server()
    # training config
    epochs = 5
    for epoch in range(1, epochs+1):
        print("\nEpoch:", epoch)
        states = np.ones([11,63,7], dtype=np.int32)
        hands = np.ones([11,7], dtype=np.int32)
        # uploade new model
        export_archive(serve_name, model, epoch, last_num_version=last_num_version)
        wait = 5
        while True:
            serve_version = max(0, epoch - last_num_version + 1) # access the previous version
            serve_version = None
            while not probe(serve_name, serve_version=serve_version):
                print("probe %s version %s :" % (serve_name, serve_version), False)
                time.sleep(wait)
            moves = send_grpc_request(states, hands, serve_name, serve_version=serve_version)
            # grpc.StatusCode.RESOURCE_EXHAUSTED : GPU OOM
            if moves is None:
                print("Waiting %d second for server to be ready..." % wait)
                print("probe:", probe(serve_name, serve_version=serve_version))
                time.sleep(wait)
                wait *= 2
            else:
                break
    # clean up
    kill_server(container_id)
    clean_archive(serve_name)
    