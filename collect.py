import _thread
from comms import read_data
from machine_learning import data_collection, sensor_data
# from pynput.keyboard import Listener, Key, KeyCode


# def register_kbd_listeners(on_move, on_quit):
#     def on_press(key):
#         if key == Key.space:
#             print('Move')
#             on_move()
#         elif key == KeyCode.from_char('q'):
#             print('Quit')
#             on_quit()
#
#     Listener(on_press=on_press).start()


if __name__ == '__main__':
    data_collection = data_collection.DataCollection('test.pb')

    try:
        # def on_quit():
        #     data_collection.save()
        #     _thread.interrupt_main()
        #
        # register_kbd_listeners(data_collection.next_move, on_quit)

        for packet in read_data.read_data():
            sensor1_datum = sensor_data.SensorDatum(packet[0:3], packet[3:6])
            sensor2_datum = sensor_data.SensorDatum(packet[6:9], packet[9:12])

            data_collection.process([sensor1_datum, sensor2_datum])

    except KeyboardInterrupt:
        data_collection.save()
        print('Done!')
