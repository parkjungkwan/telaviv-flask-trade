from simple_classification_algorithm.model import IrisModel

class IrisController:
    def __init__(self):
        self.model = IrisModel()

    @staticmethod
    def print_menu():
        print('0. 종료')
        print('1. 아이리스 데이터 출력')

    @staticmethod
    def show(param):
        print('RESULT : %s ' % param)

    
