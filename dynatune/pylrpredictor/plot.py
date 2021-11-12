import matplotlib.pyplot as plt
import numpy as np
from xlrd import open_workbook
 
def base():
    with open_workbook("C:\\Users\\t-meli\\\Documents\\muti-task-tuning\data analysis\\2-curvemodels\\singleModelTests.xlsx") as workbook:
        worksheet = workbook.sheet_by_name('all') #0.01,0.05,0.2,0.5
        for col in range(1,22):
            mname = worksheet.cell_value(0, col)
            print(mname)
            xs = []
            real_ys = []
            pre_ys = []
            for row_index in range(1, 41):
                iters = int(worksheet.cell_value(row_index, 0))
                real = float(worksheet.cell_value(row_index, 1))
                pre = float(worksheet.cell_value(row_index, col))
                xs.append(iters)
                real_ys.append(real)
                pre_ys.append(pre)
            print(xs)
            print(real_ys)
            print(pre_ys)
            plt.scatter(xs, real_ys)
            plt.scatter(xs, pre_ys)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(mname)
            # plt.show()
            plt.savefig(mname+'.png')
            plt.cla()
    
base()