from flask import Flask, render_template, request
import transalate as ts
import time
app = Flask(__name__, template_folder='/Users/ahmedmoamen/Desktop/ahmed/school/2023 spring/assembly/Risc-v Simulator/templates')


@app.route('/')
def index():
    table = ts.memory.read_page(0)
    table_1 = ts.registers.items()
    list1 = list()
    for k,v in table_1:
        list2 = list()
        list2.append(k)
        list2.append(v)
        list2.append(bin(v))
        list2.append(hex(v))
        list1.append(list2)



    return render_template('index.html', table_data_1= list1,  table_data_3=list(table))

@app.route('/execute_code', methods=['POST'])
def execute_code():
    try:
        data = request.form['code2']
        ts.readData(data.lower())
        code = request.form['code1']
        # execute the code using the appropriate method
        result = ts.readLabels(code)
        table_1 = ts.registers.items()
        list1 = list()
        for k, v in table_1:
            list2 = list()
            list2.append(k)
            list2.append(v)
            list2.append(bin(v))
            list2.append(hex(v))
            list1.append(list2)
        return render_template('index.html',table_data_1= list1,  result= result)
    except Exception as e:
        return render_template('index.html',  result="SYNATX ERROR")


