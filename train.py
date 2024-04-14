from runner import Runner
import sys

runner = Runner(config_name=sys.argv[2], load=False, render_mode="non", test=False)
runner.run()