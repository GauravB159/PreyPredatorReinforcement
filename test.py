from ppo import run
import sys

run(load = False, test = True, render_mode="human", config_name=sys.argv[2])