import re
import sys

class Page:
    def __init__(self, page_num):
        self.page_num = page_num
        self.data = [None] * 4096  # 4KB page size

    def get_page(self):
        return self.data

class Memory:
    def __init__(self):
        self.pages = {}
        self.used_cells = set()

    def read(self, addr):
        page_num, offset = divmod(addr, 4096)
        if page_num not in self.pages:
            self.pages[page_num] = Page(page_num)
        return self.pages[page_num].data[offset]

    def read_page(self, addr):
        page_num, offset = divmod(addr, 4096)
        if page_num not in self.pages:
            self.pages[page_num] = Page(page_num)
        return self.pages[page_num].get_page()

    def write(self, addr, value):
        page_num, offset = divmod(addr, 4096)
        if page_num not in self.pages:
            self.pages[page_num] = Page(page_num)
        self.pages[page_num].data[offset] = value
        self.used_cells.add((page_num, offset))

    def get_used_cells(self):
        used_cells = []
        for page_num, offset in self.used_cells:
            value = self.read(page_num * 4096 + offset)
            used_cells.append((page_num * 4096 + offset, value))
        return sorted(used_cells, key=lambda x: x[0])

class ConstantDict(dict):
    def __init__(self, constant_key, *args, **kwargs):
        self.constant_key = constant_key
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key == self.constant_key:
            print("zero not changeable")
        else:
            super().__setitem__(key, value)

regs = {
    "zero": 0,
    "ra": 0,
    "sp": 0,
    "gp": 0,
    "tp": 0,
    "t0": 0,
    "t1": 0,
    "t2": 0,
    "s0": 0,
    "s1": 0,
    "a0": 0,
    "a1": 0,
    "a2": 0,
    "a3": 0,
    "a4": 0,
    "a5": 0,
    "a6": 0,
    "a7": 0,
    "s2": 0,
    "s3": 0,
    "s4": 0,
    "s5": 0,
    "s6": 0,
    "s7": 0,
    "s8": 0,
    "s9": 0,
    "s10": 0,
    "s11": 0,
    "t3": 0,
    "t4": 0,
    "t5": 0,
    "t6": 0,
    "pc": 0
}

registers = ConstantDict("zero", regs)

memory = Memory()
sendInfo = " "
def is_integer(string):
    if string.isdigit():
        return True
    elif string[0] == '-' and string[1:].isdigit():
        return True
    else:
        return False


def add(r1, r2, r3):
    if is_integer(r3):
        registers[r1] = registers[r2] + int(r3)
    else:
        registers[r1] = registers[r2] + registers[r3]
    registers["pc"] += 4

def sub(r1, r2, r3):
    registers[r1] = registers[r2] - registers[r3]
    registers["pc"] += 4


def AND(r1,r2,r3):
    if is_integer(r3):
        registers[r1] = registers[r2] & int(r3)
    else:
        registers[r1] = registers[r2] & registers[r3]
    registers["pc"] += 4


def OR(r1,r2,r3):
    if is_integer(r3):
        registers[r1] = registers[r2] | int(r3)
    else:
        registers[r1] = registers[r2] | registers[r3]
    registers["pc"] += 4


def XOR(r1,r2,r3):
    if is_integer(r3):
        registers[r1] = registers[r2] ^ int(r3)
    else:
        registers[r1] = registers[r2] ^ registers[r3]
    registers["pc"] += 4


def separateOffset(input_str):
    match = re.match(r'^(\d+)\((\w+)\)$', input_str)

    if match:
        num = int(match.group(1))
        reg = match.group(2)
        result = [int(num), reg]
        return result
# add more functions here to handle other instructions
def LBU(r1, r2):
        memoryOff = separateOffset(r2)
        location = memoryOff[0]+registers[memoryOff[1]]
        registers[r1] = int(memory.read(location), 2)
        registers["pc"] += 4

def LHU(r1, r2):
    memoryOff = separateOffset(r2)
    location = memoryOff[0]+registers[memoryOff[1]]
    binary = memory.read(location) + memory.read(location+1)[2:]
    registers[r1] = int(binary, 2)
    registers["pc"] += 4


def LWU(r1, r2):
    if r2 in data:
        registers[r1] = abs(data[r2])
        registers["pc"] += 4
    else:
        memoryOff = separateOffset(r2)
        location = memoryOff[0]+registers[memoryOff[1]]
        binary = memory.read(location) + memory.read(location+1)[2:] + memory.read(location+2)[2:] + memory.read(location+3)[2:]
        registers[r1] = int(binary, 2)
        registers["pc"] += 4


def LDU(r1, r2):
    memoryOff = separateOffset(r2)
    location = memoryOff[0] + registers[memoryOff[1]]
    binary = memory.read(location) + memory.read(location + 1)[2:] + memory.read(location + 2)[2:] + memory.read(location + 3)[2:] + memory.read(location + 4)[2:] + memory.read(location + 5)[2:] + memory.read(location + 6)[2:] + memory.read(location + 7)[2:]
    registers[r1] = int(binary, 2)
    registers["pc"] += 4


def binary_to_signed_int(bin_str):
    # Check if the binary string is negative
    is_negative = False
    if bin_str[2] == '1':
        is_negative = True

    # Convert the binary string to an integer
    int_val = int(bin_str, 2)

    # If the binary string is negative, convert it to a signed integer
    if is_negative:
        # Flip all the bits
        flipped_bits = ''.join(['0' if b == '1' else '1' for b in bin_str])
        # Add 1 to the flipped bits
        int_val = -(int('0b' + flipped_bits, 2) + 1)

    return int_val

def rshift(val, n): return (val % 0x100000000) >> n

def LB(r1, r2):
    memoryOff = separateOffset(r2)
    location = memoryOff[0]+registers[memoryOff[1]]
    registers[r1] = binary_to_signed_int(memory.read(location)[2:])
    registers["pc"] += 4

def LH(r1, r2):
    memoryOff = separateOffset(r2)
    location = memoryOff[0]+registers[memoryOff[1]]
    binary = memory.read(location) + memory.read(location+1)[2:]
    registers[r1] = binary_to_signed_int(binary[2:])
    registers["pc"] += 4


def LW(r1, r2):
    memoryOff = separateOffset(r2)
    location = memoryOff[0]+registers[memoryOff[1]]
    binary = memory.read(location) + memory.read(location+1)[2:] + memory.read(location+2)[2:] + memory.read(location+3)[2:]
    registers[r1] = binary_to_signed_int(binary[2:])
    registers["pc"] += 4


def LD(r1, r2):
    memoryOff = separateOffset(r2)
    location = memoryOff[0] + registers[memoryOff[1]]
    binary = memory.read(location) + memory.read(location + 1)[2:] + memory.read(location + 2)[2:] + memory.read(location + 3)[2:] + memory.read(location + 4)[2:] + memory.read(location + 5)[2:] + memory.read(location + 6)[2:] + memory.read(location + 7)[2:]
    registers[r1] = binary_to_signed_int(binary[2:])
    registers["pc"] += 4


def sll(r1, r2, r3):
    if is_integer(r3):
        registers[r1] = registers[r2] << int(r3)
    else:
        registers[r1] = registers[r2] << registers[r3]
    registers["pc"] += 4


def srl(r1, r2, r3):
    if is_integer(r3):
        registers[r1] = rshift(registers[r2], int(r3))
    else:
        registers[r1] = rshift(registers[r2], registers[r3])
    registers["pc"] += 4

def sra(r1, r2, r3):
    if is_integer(r3):
        registers[r1] = registers[r2] >> int(r3)
    else:
        registers[r1] = registers[r2] >> registers[r3]
    registers["pc"] += 4


def beq(r1, r2, label):
    if registers[r1] == registers[r2]:
         registers["pc"] = label_addressmap[label]
    else:
        registers["pc"] += 4


def bne(r1, r2, label):
    if registers[r1] != registers[r2]:
        registers["pc"] = label_addressmap[label]
    else:
        registers["pc"] += 4


def bge(r1, r2, label):
    if registers[r1] >= registers[r2]:
        registers["pc"] = label_addressmap[label]
    else:
        registers["pc"] += 4


def bgeu(r1, r2, label):
    if abs(registers[r1]) >= abs(registers[r2]):
        registers["pc"] = label_addressmap[label]
    else:
        registers["pc"] += 4


def blt(r1, r2, label):
    if registers[r1] < registers[r2]:
        registers["pc"] = label_addressmap[label]
    else:
        registers["pc"] += 4


def bltu(r1, r2, label):
    if abs(registers[r1]) < abs(registers[r2]):
        registers["pc"] = label_addressmap[label]
    else:
        registers["pc"] += 4


def slt(r1, r2, r3):
    if r2 in registers:
        val1 = registers[r2]
        if is_integer(r3):
            val2 = int(r3)
        else:
            if r3 in registers:
                val2 = registers[r3]
            else:
                print("Invalid register name(s) in instruction")
                return
        if val1 < val2:
            registers[r1] = 1
        else:
            registers[r1] = 0
    else:
        print("Invalid register name(s) in instruction")
    registers["pc"] += 4

def sltu(r1, r2, r3):
    if r2 in registers:
        val1 = abs(registers[r2])
        try: #check whether the third argument is a string that can be converted to an integer
            val2 = abs(int(r3))
        except Exception:
            if r3 in registers:
                val2 = abs(registers[r3])
            else:
                print("Invalid register name(s) in instruction")
                return
        if val1 < val2:
            registers[r1] = 1
        else:
            registers[r1] = 0
        registers["pc"] += 4

def lui(rd, imm20): #work in decimals
    if rd not in registers:
        print("Invalid register name")
        return
    if not 0 <= int(imm20) <= 0xfffff:
        print("Invalid immediate value")
        return
    registers[rd] = int(imm20) << 12
    registers["pc"] += 4

def twos_complement(bin_str: str) -> str:
    # Flip all the bits
    flipped = ''.join(['0' if b == '1' else '1' for b in bin_str])

    # Add 1 to the flipped bits
    carry = '1'
    result = ''
    for b in reversed(flipped):
        if b == '0' and carry == '0':
            result = '0' + result
            carry = '0'
        elif b == '1' and carry == '0':
            result = '1' + result
            carry = '0'
        elif b == '0' and carry == '1':
            result = '1' + result
            carry = '0'
        elif b == '1' and carry == '1':
            result = '0' + result
            carry = '1'

    # If carry is still 1, add it to the result
    if carry == '1':
        result = '1' + result

    return result

def int_to_signed_bin(num: int, num_bits: int) -> str:
    # Check if number is negative
    is_negative = num < 0

    # Convert to binary string with padding
    bin_str = format(abs(num), f'0{num_bits}b')

    # If negative, take 2's complement
    if is_negative:
        bin_str = twos_complement(bin_str)

    return bin_str

def sb(rs2, offset_rs1):
    # separate offset from base register
    offset, rs1 = separateOffset(offset_rs1)

    # calculate address
    addr = registers[rs1] + offset

    # write value to memory
    value = bin(registers[rs2])
    if len(value) > 10:
        print("too big")
        return

    if len(value[2:]) < 8:
        value = value[2:].zfill(8)
    memory.write(addr, "0b"+value)
    registers["pc"] += 4

def sh(rs, offset_r2):
    offset, rs2 = separateOffset(offset_r2)
    addr = offset + registers[rs2]
    value = int_to_signed_bin(registers[rs], 16)
    if len(value) > 18:
        print("too big")
        return
    memory.write(addr, "0b" + value[0:8])
    memory.write(addr+1, "0b" + value[8:])
    registers["pc"] += 4

def sw(rs, offset_r2):
    offset, rs2 = separateOffset(offset_r2)
    addr = offset + registers[rs2]
    value = int_to_signed_bin(registers[rs], 32)
    if len(value) > 34:
        print("too big")
        return
    print(value)
    memory.write(addr, "0b" + value[0:8])
    memory.write(addr+1, "0b" + value[8:16])
    memory.write(addr+2, "0b" + value[16:24])
    memory.write(addr + 3, "0b" + value[24:])
    registers["pc"] += 4

def sd(rs, offset_r2):
    offset, rs2 = separateOffset(offset_r2)
    addr = offset + registers[rs2]
    value = int_to_signed_bin(registers[rs], 64)
    if len(value) > 66:
        print("too big")
        return
    memory.write(addr, "0b" + value[0:8])
    memory.write(addr + 1, "0b" + value[8:16])
    memory.write(addr + 2, "0b" + value[16:24])
    memory.write(addr + 3, "0b" + value[24:32])
    memory.write(addr+4, "0b" + value[32:40])
    memory.write(addr+5, "0b" + value[40:48])
    memory.write(addr+6, "0b" + value[48:56])
    memory.write(addr + 7, "0b" + value[56:])
    registers["pc"] += 4


def jal(r1, imm):
    registers[r1] = registers["pc"] + 4
    if isinstance(imm, int):
        registers["pc"] += imm
    else:
        registers["pc"] = label_addressmap[imm]


def jalr(r1, imm_r2):
    imm, r2 = separateOffset(imm_r2)
    location = registers[r2] + imm
    registers[r1] = registers["pc"] + 4
    registers["pc"] = location

def li(r1, imm):
    registers[r1] = int(imm);
    registers["pc"] += 4

label_addressmap = {
}

instruction_map = {
    "add": add,
    "addi": add,
    "sub": sub,
    "and": AND,
    "or": OR,
    "xor": XOR,
    "andi": AND,
    "ori": OR,
    "xori": XOR,
    "lb": LB,
    "lh": LH,
    "lw": LW,
    "ld": LD,
    "lbu": LBU,
    "lhu": LHU,
    "lwu": LWU,
    "ldu": LDU,
    "sll": sll,
    "srl": srl,
    "sra": sra,
    "slli": sll,
    "srli": srl,
    "srai": sra,
    "beq": beq,
    "bne": bne,
    "bge": bge,
    "bgeu": bgeu,
    "blt": blt,
    "bltu": bltu,
    "slt": slt,
    "slti": slt,
    "sltu": sltu,
    "sltiu": sltu,
    "lui": lui,
    "sb": sb,
    "sh": sh,
    "sw": sw,
    "sd": sd,
    "jal": jal,
    "jalr": jalr,
    "li": li
}

codeaddress = {}

"""def translate_instruction(instruction):
    # split the instruction into tokens
    instruction = instruction.lower()
    instruction = instruction.replace(",", "")
    tokens = instruction.split()


    # extract the operation and register names
    op = tokens[0]
    # look up the function for the given operation
    if op in instruction_map:
        operation_func = instruction_map[op]
        try:
             return operation_func(tokens[1], tokens[2], tokens[3])
        except Exception:
            return operation_func(tokens[1], tokens[2])

    else:
        return "Unsupported instruction"""


def readLabels(textFile):
    temp =""
    instructions = textFile.split("\n")
    counter = registers["pc"]
    for instruction in instructions:
        instruction = instruction.lower()
        label_instr = instruction.split(":")
        if len(label_instr) == 2:
            label = label_instr[0].strip(":")
            label_addressmap[label] = counter
        codeaddress[counter] = instruction
        counter += 4
    size = len(codeaddress)
    pcIntial = registers["pc"]
    while pcIntial + size*4 > registers["pc"]:
        instruction = translate_instruction(codeaddress[registers["pc"]])
        temp += "\n" + instruction + "\n" + str(registers) + "\n" + str(memory.get_used_cells()) + "\n"
    return temp




def readData(text):
    datas = text.split("\n")
    for var in datas:
        var = var.split(",")
        try:
            value = int(var[1])
            value = int_to_signed_bin(value, 32)
            addr = int(var[0])
            memory.write(addr, "0b" + value[0:8])
            memory.write(addr + 1, "0b" + value[8:16])
            memory.write(addr + 2, "0b" + value[16:24])
            memory.write(addr + 3, "0b" + value[24:])
        except Exception:
            print("value not integer")



def translate_instruction(instruction):
    # split the instruction into tokens
    instruction = instruction.lower()
    instruction = instruction.replace(",", "")
    label_instr = instruction.split(":")

    if len(label_instr) == 2:
        instruction = label_instr[1].strip()



    tokens = instruction.split()
    # extract the operation and register names
    op = tokens[0]
    if op == "ecall":
        registers["pc"] = sys.maxsize
    # look up the function for the given operation
    if op in instruction_map:
        operation_func = instruction_map[op]
        try:
             operation_func(tokens[1], tokens[2], tokens[3])
             return instruction
        except Exception:
            try:
             operation_func(tokens[1], tokens[2])
             return instruction
            except Exception as e:
                print(e)
                registers["pc"] = sys.maxsize
    else:
        return "Unsupported instruction"




"""addi a2, zero, 3
addi a3, zero, 3
bne a2, a3, Else
add a0, a1, a2
beq zero, zero, Exit 
Else: sub a0, a1, a2
Exit: addi s0, s0, 1"""




#test case
"""fact:addi sp, sp, -8
sw ra, 4(sp)
sw a0, 0(sp)
slti t0, a0, 1
beq t0, zero, L1
addi a0, zero, 1
addi sp, sp, 8
jalr zero, 0(ra)
L1: addi a0, a0, -1
jal ra, fact
addi t0, a0, 0
lw a0, 0(sp)
lw ra, 4(sp)
addi sp, sp, 8
jalr zero, 0(ra)"""






