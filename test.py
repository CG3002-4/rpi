from recv_data import recv_data

if __name__ == '__main__':
    print('Testing...')

    for unpacked_data in recv_data():
        print(unpacked_data)
