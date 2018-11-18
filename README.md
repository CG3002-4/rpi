# Dance Dance

A wearable system that predicts the dance move you're performing.

## Workflow

### Getting Started

1. Plug a battery pack into the system.
2. Connect the RPi to a monitor, mouse and keyboard.
3. Set up WiFi for the RPi.
4. (Optional) Create a Telegram chat and set the `chat_id` in `telegram_bot.py` to that of your chat. This will enable the RPi to text you its IP address on startup so that the peripherals are not needed again.
5. Change working directory in the RPi to `~/Documents/CG3002/CG3002-4/rpi` and update the git repository to the latest version. Alternatively, clone the repository at [github/CG3002-4/rpi](#https://github.com/CG3002-4/rpi)
5. Run `python3 test.py` to test if the RPi is receiving data from the Arduino. (You may have to toggle the `HANDSHAKING` flag in `clientconnect.py`)
6. Run `pip3 install -r requirements.txt` on your computer to install the required packages.

### Data Collection

1. Run `python3 rpi_collect.py <data_folder_name>/<experiment_name> <move_label>`. For example, _John_ was performing the _mermaid_ move, for which the label is _8_ and we wish to store all our data in a folder called _data_, then the command could be: `python3 rpi_collect.py data/johnmermaid 8`
2. Send a KeyboardInterrupt to the program when done dancing. This is `Ctrl-C` on Windows and `Cmd-C` on Mac OS.
3. Transfer the data to your computer if you wish to train there. For example, the data could be committed and pushed to GitHub, then pulled onto your computer.

### Modelling

The file `pipeline.py` handles all operations related to modelling. Run `python3 pipeline.py -h` to see a list of options. These are also explained below:
* There are two ways to load data for feature extraction. The first uses the `--all-exp` option, which loads data from all folders under `data/`. The second uses the `--exp-names` option, which takes in a list of globs and loads data from all experiments under `data/` matching these globs. (To change the default data folder, change the `DATA_FOLDER` variable in `pipeline.py`)
* To extract features from, for example, all the experiments and save them in `features.csv`, run `python3 pipeline.py save-features --all-exp --data-file features`.
* To run cross validation with, for example, generated features, run `python3 pipeline.py cross --load-file features`.
* To train a model with, for example, only those experiments containing _john_ and save the model to `trained.pb`, run `python3 pipeline.py train --exp-names *john* --model-file trained`

Note that the model must be trained on the RPi due to compatibility issues between 32-bit and 64-bit python. It is useful to extract features on a computer and transfer these over to the RPi for training since the RPi's computational power is limited.

### Running the device

Run `python3 rpi_realtime.py <model_file_name> <server_ip> <port> server` if you wish the device to send predictions to the server. If not, leave out the last three arguments. Leaving out only the last argument will also have the same effect.

Prediction probabilities will be printed out to the terminal.
