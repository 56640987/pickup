import serial
from time import sleep

ser = serial.Serial()

def port_open():
    ser.port = '/dev/ttyCH343USB0'
    ser.baudrate = 115200
    ser.bytesize = 8
    ser.stopbits = 1
    ser.parity = 'N'
    ser.open()
    if(ser.isOpen()):
        print("串口打开成功1")
    else:
        print("串口打开失败1")

def port_close():
    ser.close()
    if(ser.isOpen()):
        print("串口关闭失败2")
    else:
        print("串口关闭成功2")

def port_send_open():
    if (ser.isOpen()):
        data1 = [0x7b, 0x01, 0x02, 0x00, 0x20, 0x49, 0x20, 0x00, 0xc8, 0xf9, 0x7d]
        ser.write(data1)
        print("发送成功1")
    else:
        print("发送失败1")

def port_send_close():
    if (ser.isOpen()):
        for i in range(2): 
            data2 = [0x7b, 0x01, 0x02, 0x01, 0x20, 0x49, 0x20, 0x00, 0xc8, 0xf8, 0x7d] 
            ser.write(data2)
            sleep(0.5)
        print("发送成功2")
    else: 
        print("发送失败2")

if __name__ == '__main__':
    port_open()
    port_send_close()
    # port_send_close()
    port_send_close()
    # port_send_open()
    port_close()
    
