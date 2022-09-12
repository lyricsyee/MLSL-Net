import os
import datetime
import shutil

import logging 
from logging import Formatter
from logging.handlers import RotatingFileHandler

import json 
# import demjson
from pprint import pprint
from glob import glob 

from utils.dirs import create_dirs
from utils.easydict import EasyDict

def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**7, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_error_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_error_file_handler.setLevel(logging.WARNING)
    exp_error_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_error_file_handler)

def get_config_from_json(json_file):
    '''
    Get config from a json file.
    :param json_file: the path of config file
    :return: config(namespace), config(dictionary)
    '''

    # parse the configuration
    with open(json_file, 'r') as f:
        try:
            # import pdb; pdb.set_trace()
            config_dict = json.load(f)
            config = EasyDict(config_dict)
            return config, config_dict
        except:
            print('INVALID JSON file format.. Please check.')
            exit(-1)


def process_config(json_file):
    '''
    Get the json file. Processing it to be accessible.
    ...

    :param json_file: the path of the config file
    :return: config, object(namespace)
    '''
    config, _ = get_config_from_json(json_file)
    print('The Configuration of the experiments .. ')
    # pprint(config)

    # making sure that you have provided the exp_name.
    try:
        print(" ****************************************************** ")
        print(" ******** The experiment name is {} !~~".format(config.exp_name))
        print(" ****************************************************** ")
    except:
        print("ERROR!!!...Please provide the exp_name in json file: {}"
              .format(json_file))
        exit(-1)

    results_root = config.results_root
    agent = config.agent
    # create some important directories to be used for that experiments. 
    try:
        # results_root = config.results_root
        # cur_exp_root = os.path.join(
        #     results_root, config.agent, config.exp_name 
        #     # '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
        # )
        # import pdb; pdb.set_trace()
        exps_names = glob(os.path.join(
            results_root, agent, config.exp_name.format('*')))
        finish_len = len(exps_names)
        cur_exp_root = os.path.join(
            results_root, agent, config.exp_name.format(finish_len + 1))

        print('ROOT DIR: {}'.format(cur_exp_root))
    except:
        print('ERROR!!..Lack of some attributes!')
        exit(-1)


    config.summary_dir = os.path.join(cur_exp_root, 'summaries/')
    config.checkpoint_dir = os.path.join(cur_exp_root, 'checkpoints/')
    config.out_dir = os.path.join(cur_exp_root, 'out/')
    config.log_dir = os.path.join(cur_exp_root, 'logs/')
    create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir])

    # save config parameters to result folder
    shutil.copyfile(json_file, os.path.join(cur_exp_root, 'params.json'))

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info('Hi, This is root.')
    logging.getLogger().info('After the configurations are successfully processed and dirs are created!')
    logging.getLogger().info('The pipeline of the project will begin now.')

    return config











