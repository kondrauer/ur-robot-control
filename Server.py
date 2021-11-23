import socket

class Server:
	
	def __init__(self, family = socket.AF_INET, type = socket.SOCK_STREAM, ip: str = '10.83.2.1', port: int = 2000) -> None:
		
		self.serverSocket: socket.socket = socket.socket(family = family, type = type) 
		self.ip = socket.gethostbyname(socket.gethostname())
		print(self.ip)
		self.serverAddress: tuple = (ip, port)

		self.serverSocket.bind(self.serverAddress)
		self.commSocket = None
		self.commAddress = None
		
	def __del__(self) -> None:
		
		self.serverSocket.close()
		print('Connection closed')
		
	def establishConnection(self):
		
		# try ergänzen
		self.serverSocket.listen(1)
		commSocket, commAddress = self.serverSocket.accept()
		
		print(f' Connected to {commAddress[0]}:{commAddress[1]}')
		
		return commSocket, commAddress
		
	def closeConnection(self) -> None:
		
		self.serverSocket.close()
		
		self.commSocket = None
		self.commAddress = None
		
		print('Connection closed')
		
	def sendData(self, data: str) -> None:
	
		self.commSocket.send(data.encode())
		
	def receiveData(self, bytes: int) -> str:
	
		return self.commSocket.recv(bytes).decode()
		
