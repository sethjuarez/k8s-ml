import os
import azureml
import tensorflow as tf
from subprocess import call
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication 

def mount_blob_storage(container, path, temp_path):
    cmds = ["blobfuse", "{}", "--container-name={}", "--tmp-path={}"]
    cmds[1] = cmds[1].format(path)
    cmds[2] = cmds[2].format(container)
    cmds[3] = cmds[3].format(temp_path)
    call(cmds)
    return path

def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)

def run(model_path, model_name):
    info('Azure ML SDK Version: {}'.format(azureml.core.VERSION))

    print(model_path, model_name)

    auth_args = {
        'tenant_id': os.environ['TENANT_ID'],
        'service_principal_id': os.environ['SERVICE_PRINCIPAL_ID'],
        'service_principal_password': os.environ['SERVICE_PRINCIPAL_PASSWORD']
    }

    ws_args = {
        'auth': ServicePrincipalAuthentication(**auth_args),
        'subscription_id': os.environ['SUBSCRIPTION_ID'],
        'resource_group': os.environ['RESOURCE_GROUP']
    }

    ws = Workspace.get(os.environ['WORKSPACE_NAME'], **ws_args)

    print(ws.get_details())

if __name__ == "__main__":
    # argparse stuff for model path and model name
    
    info('Using TensorFlow v.{}'.format(tf.__version__))

    # ENV set, we are mounting blob storage
    if 'BASE_PATH' in os.environ:
        print('Mounting blob storage')
        base_path = mount_blob_storage(os.environ['AZURE_STORAGE_CONTAINER'], 
                                        os.environ['BASE_PATH'], 
                                        os.environ['TEMP_PATH'])
    else:
        base_path = '..'

    run('PATH', 'NAME')
