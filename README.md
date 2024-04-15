## Setup
1. Clone the repository and fetch larger files from git lfs if needed
2. Create a virtual environment and install all requirements from the file
3. A VSCode launch.json has been provided to easily train and test existing configurations
4. Training can also be run using the following command as an example: ```python main.py --config single_prey --load false --render_mode no --mode train```
5. Training can be continued by setting the load argument to true 
6. Testing can also be run using the following command as an example: ```python main.py --config single_prey --load true --render_mode human --mode test```