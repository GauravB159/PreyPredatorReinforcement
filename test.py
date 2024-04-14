from runner import Runner
import sys

runner = Runner(config_name=sys.argv[2], load=True, render_mode="human", test=True)
runner.run()