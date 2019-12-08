# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:20:35 2019

parse config file

@author: as 
"""
import io
import os
import yaml
import shutil


def load_config_file(yaml_file):
    file = io.open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    print("The configuration is as follows:")
    print(file_data)
    file.close()
    config_dict = yaml.load(file_data)
    return config_dict


def mkdir_if_nonexist(check_dir, raise_error=False):
    """
    check the input directory, if not exist, build it.
    Args:
        check_dir: the direcory to check
        raise_error: if exist, raise error or ignore and remove it
    """
    if not os.path.exists(check_dir):
        os.mkdir(check_dir)
    else:
        if raise_error:
            raise RuntimeError("Warning! {} has exist! please check".format(check_dir))
        else:
            print("Warning! {} has exist! remove it now!".format(check_dir))
            shutil.rmtree(check_dir)
            os.mkdir(check_dir)



def import_model_by_networkname(network_name):
    import_str = ""
    if network_name.startswith('efficient'):
        import_str = 'from net.efficient_net.efficient_net_builder import EfficientNetModel as Model'
    elif network_name.startswith('resnet-v1'):
        import_str = 'from net.resnet_slim.resnet_v1_builder import ResNetModel as Model'
    elif network_name.startswith('resnet-v2'):
        import_str = 'from net.resnet_slim.resnet_v2_builder import ResNetModel as Model'

    #from net.resnet.resnet_v2_builder import ResNetModel as Model
    #from net.se_resnet.se_resnet_v2_builder import SEResNetModel as Model
    #from net.se_resnet.se_resnet_v2_builder import SEResNetModel as Model
    return import_str



def import_ckpt_predictor_by_networkname(network_name):
    import_str = ""
    if network_name.startswith('efficient'):
        import_str = 'from net.efficient_net.efficient_net_ckpt_predictor import EfficientNet_Predictor'
    else:
        import_str = 'pass'
    return import_str
