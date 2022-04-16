import socket

import logging

class Server:
	"""Small wrapper class for built-in socket module
	
	Attributes:
		serverSocket(socket.socket): socket object for low lever network interfacing
		ip(str): ip as string
		serverAddress(tuple): tuple of ip as string and port as int
		
		commSocket(socket.socket): socket object for communicating with robot
		commAddress(tuple): address of communication socket
		
	"""	
	
	def __init__(self, family = socket.AF_INET, type = socket.SOCK_STREAM, ip: str = '10.83.2.1', port: int = 2000) -> None:
		"""Constructor for Server
		
		Args:
			family: determines connection type, default is ipv4
			type: determines protocol type, default is tcp
			ip(str): ip as string, if no ip is given constructor tries to get one
			port(int): port as int
			
		"""
		
		self.serverSocket: socket.socket = socket.socket(family = family, type = type) 
		
		if not ip:
			self.ip = socket.gethostbyname(socket.gethostname())
		else:
			self.ip = ip
		
		self.port = port
		
		self.serverAddress: tuple = (self.ip, self.port)
		self.serverSocket.bind(self.serverAddress)
		logging.info(f'Socket bound to: {self.ip}:{self.port}')
		
		self.commSocket: socket.socket = None
		self.commAddress: tuple = None
		
	def __del__(self) -> None:
		"""Destructor for Server
		
		Tries to close connection if not already done
		"""
		self.closeConnection()
		
	def establishConnection(self):
		"""Listens for and establishes connections
		
		Returns:
			socket.socket: socket for communication
			tuple: ip as str and port of communicating client 
		"""
		
		#TODO: try ergänzen
		self.serverSocket.listen(1)
		
		while True:
			commSocket, commAddress = self.serverSocket.accept()
			
			if commSocket and commAddress:
				self.commSocket = commSocket
				self.commAddress = commAddress
				break
		
		logging.info(f'Connected to {commAddress[0]}:{commAddress[1]}')
		
		return commSocket, commAddress
		
	def closeConnection(self) -> None:
		"""Closes connection to client

		"""
		if self.commSocket and self.commAddress:
				
			self.serverSocket.close()
		
			self.commSocket = None
			self.commAddress = None
		
			print('Connection closed')
		
		else:
			print('No connection established, cant be closed!')
		
	def sendData(self, data: str) -> None:
		"""Encodes and sends data to connected client.
		
		Args:
			data(str): data to send

		"""
		if self.commSocket and self.commAddress:
		
			self.commSocket.send(data.encode())
			logging.info('Sending data')
			
		else:
			logging.error('No connection established, sending not possible!')
			
	def receiveData(self, bytes: int) -> str:
		"""Receives and decodes data from connected client.
		
		Returns:
			str: received bytes decoded, None if no connection

		"""
		if self.commSocket and self.commAddress:
			logging.info('Receiving data')
			return self.commSocket.recv(bytes).decode()
		else:
			logging.error('No connection established, receiving not possible!')
			return None

import numpy as np

if __name__ == '__main__':
    
	logging.getLogger().setLevel(logging.INFO)
	server = Server()
	
	commSocket, commAddress = server.establishConnection()

	data = server.receiveData(1024)
	
	data = data[2:len(data)-1]
	np_Temp = np.fromstring(data, dtype=np.float64, sep=",")
	np_Tvec = np_Temp[0:3].copy().reshape(3,1)
	np_Rvec = np_Temp[3:6].copy().reshape(3,1)
	print(np_Temp)
	server.sendData('1')
	
	data = server.receiveData(1024)
	print(data)
	server.sendData('1')
	server.closeConnection()
