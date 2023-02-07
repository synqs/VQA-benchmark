from qiskit import BasicAer
from general.myTypes import BaseBackend




def get_backend(hardware: str) -> BaseBackend:
	if hardware[:5] == 'ibmq_':
		from qiskit import IBMQ
		IBMQ.load_account()
		provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
	return BasicAer.get_backend(hardware)
