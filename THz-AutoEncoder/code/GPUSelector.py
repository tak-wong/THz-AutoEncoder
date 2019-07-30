#   Distribution code Version 1.0 -- 24/07/2019 by Tak Ming Wong Copyright 2019, University of Siegen
#
#   The Code is created based on the method described in the following paper 
#   [1] "Training Auto-encoder-based Optimizers for Terahertz Image Reconstruction", T.M. Wong, M. Kahl, P. Haring Bolivar, A. Kolb, M. Moeller, 
#   German Conference on Pattern Recognition (GCPR), 2019.
#   
#   If you use this code in your scientific publication, please cite the mentioned paper.
#   The code and the algorithm are for non-comercial use only.

from subprocess import Popen, PIPE
import os

class GPUInfo():
    def __init__(self):
        self.GPUid=0
        self.name=""
        self.MemUtil=0
        self.Temp=0
        self.GPUUtil=0

    def getGPUs(self):
        try:
            p = Popen(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.total,memory.free,name,temperature.gpu", "--format=csv,noheader,nounits"], stdout=PIPE)
            stdout, _ = p.communicate()
        except:
            return []
        
        print("\nSearching GPU with max free Memory\n")

        MemUtil=-1
        output = stdout.decode('UTF-8')
        lines = output.split(os.linesep)
        numDevices = len(lines)-1

        print("ID\tGPU\t\t\tFree Memory\tUtilization")
        for g in range(numDevices):
            line = lines[g]
            vals = line.split(', ')
            for i in range(5):
                if (i == 0):
                    deviceIds = int(vals[i])
                elif (i == 1):
                    gpuUtil = int(vals[i])
                elif (i == 3):
                    memFree = int(vals[i])
                elif (i == 4):
                    gpu_name = vals[i]
                   
            print("{}\t{}\t{}\tMB\t{}%".format(deviceIds,gpu_name,memFree,gpuUtil))

            if(memFree > MemUtil):
                MemUtil = memFree
                IDtoUSE = deviceIds
                name = gpu_name
        
        self.GPUid=IDtoUSE
        self.name=name
        self.MemUtil=MemUtil       

        print("\nSelect GPU {} {}".format(IDtoUSE,name))

        return IDtoUSE

    def getGPUName(self):
        return self.name

    def getMemUtil(self):
        return self.MemUtil

    def getTemp(self):
        return self.Temp

    def getGPUUtil(self):
        return self.GPUUtil

    def updateInfo(self):
        try:
            p = Popen(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.total,memory.free,name,temperature.gpu","-i "+str(self.GPUid), "--format=csv,noheader,nounits"], stdout=PIPE)
            stdout, _ = p.communicate()
        except:
            return []

        output = stdout.decode('UTF-8')
        lines = output.split(os.linesep)
        numDevices = len(lines)-1

        for g in range(numDevices):
            line = lines[self.GPUid]
            vals = line.split(', ')
            for i in range(6):
                if (i == 1):
                    gpuUtil = int(vals[i])
                elif (i == 2):
                    memTotal = int(vals[i])
                elif (i == 3):
                    memFree = int(vals[i])
                elif (i == 4):
                    gpu_name = vals[i]
                elif (i == 5):
                    temp_gpu = float(vals[i])
            
        self.MemUtil=memTotal-memFree  
        self.Temp = temp_gpu 
        self.GPUUtil= gpuUtil
        self.name = gpu_name


