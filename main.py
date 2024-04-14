from runner import Runner
import argparse

parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--config")
parser.add_argument("-l", "--load")
parser.add_argument("-r", "--render_mode")
parser.add_argument("-m", "--mode")
args = vars(parser.parse_args())
runner = Runner(config_name=args['config'], load=args['load'] == "true", render_mode = args['render_mode'], test=args['mode'] == "test")
runner.run()