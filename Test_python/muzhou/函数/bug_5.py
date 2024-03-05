import traceback
try:
    print('__________')
    print(1/0)
except:
    traceback.print_exc()