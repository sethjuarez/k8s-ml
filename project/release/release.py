import os
import json
import time
import azureml
from dotenv import load_dotenv
from azureml.core.model import Model
from azureml.core.image import ContainerImage, Image
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import Webservice, AciWebservice, AksWebservice




def get_workspace():
    try:
        
        ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
        ws.write_config()
        return ws
    except:
        return None

def save_model(file_path):



if __name__ == "__main__":
    load_dotenv(verbose=True)

    

