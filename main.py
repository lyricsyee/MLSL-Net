import argparse
from tqdm import tqdm
from utils.config import *
from agents import *



def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        'config',
        metavar='config_json_file',
        default=None,
        help='The Configuration file in json format!'
    )
    args = parser.parse_args()

    # parse the config json file
    config = process_config(args.config)

    # Create the Agent and pass all the configuration to it then run it.
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
