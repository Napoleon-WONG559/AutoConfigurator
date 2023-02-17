import pandas as pd
import re

#map text configuration of graphic card to numerical label using rank_type list
#return dict in form of {label1:[text configuration1-1, ...], label2:[text configuration2-1, ...], ...}
def map_graphicard(graphicard):
    #mapping standard, 'other' type is for remaining text configuration that doesn't match the existing type rules
    rank_type={1:['NVIDIA GeForce GTX 1070', 'GTX 1070', 'NVIDIA GeForce GTX 1060','NVIDIA GeForce GTX 1050', 'GTX 1050'],
          2:['AMD Radeon R7','radeon r5', 'Radeon R5', 'Radeon R4', 'NVIDIA GeForce 940MX', 'Intel Iris Plus', 'Intel UHD Graphics 620'],
           3:['Intel', 'Intel HD', 'intel', 'AMD Radeon R2'], 4:['Integrated', 'integrated'], 5:['other']}
    
    #save the text configuration that match the standard in corresponding label
    rank_dic={}
    
    counter={}
    for item in graphicard:
        counter[item]=0
    for key,val in rank_type.items():
        tmp=[]
        for item in graphicard:
            flag=0
            for i in val:
                if(re.findall(i,str(item))!=[]):
                    flag=1
                    break
            if(flag!=0):
                counter[item]+=1
                tmp.append(item)
        rank_dic[key]=tmp
    for key,val in counter.items():
        if(val>1):
            print(key,': ',val)
    return rank_dic

#return cpu list sorted by processing frequecy
def map_cpu(cpu):
    cpu_list=list(cpu)
    cpu_dic={}
    for item in cpu_list:
        #print(item)
        if(item==8032):
            cpu_dic[item]=101
            continue
        if(item[0].isdigit()):
            cpu_dic[item]=int(item[0])
        else:
            cpu_dic[item]=100
    sort_cpu=sorted(cpu_dic.items(), key=lambda x:x[1])
    res=[]
    for item in sort_cpu:
        res.append(item[0])
    return res

#return ram list sorted by capacity/size
def map_ram(ram):
    ram_list=list(ram)
    ram_dic={}
    for item in ram_list:
        if(item[0].isdigit()):
            ram_dic[item]=int(item[:2])
        else:
            ram_dic[item]=100
    sort_ram=sorted(ram_dic.items(), key=lambda x:x[1])
    res=[]
    for item in sort_ram:
        res.append(item[0])
    return res

#return disk list sorted by capacity/size
def map_disk(hardisk):
    disk_list=list(hardisk)
    disk_dic={}
    for item in disk_list:
        if('128' in item):
            disk_dic[item]=128
        elif('256' in item):
            disk_dic[item]=256
        elif('500' in item):
            disk_dic[item]=500
        elif('512' in item):
            disk_dic[item]=512
        elif('1 TB' in item or '1000' in item or '1024' in item or '2 TB' in item):
            disk_dic[item]=1000
        else:
            disk_dic[item]=0
    sort_disk=sorted(disk_dic.items(), key=lambda x:x[1])
    res=[]
    for item in sort_disk:
        res.append(item[0])
    return res

if __name__ == '__main__':
    label_file=pd.read_excel('labels for the laptop (specifications).xlsx', index_col=0, header=0)

    #extract all text configuration information from the file for each label(screen, cpu, ram, hard disk, graphic card)
    screen=[]
    cpu=[]
    ram=[]
    hardisk=[]
    graphicard=[]
    for i in label_file.keys():
        screen.append(label_file[i][0])
        cpu.append(label_file[i][1])
        ram.append(label_file[i][2])
        hardisk.append(label_file[i][3])
        graphicard.append(label_file[i][4])
    screen=set(screen)
    cpu=set(cpu)
    ram=set(ram)
    hardisk=set(hardisk)
    graphicard=set(graphicard)

    #obtain the mapped/sorted result
    graphicard_dic=map_graphicard(graphicard)
    cpu_sort_list=map_cpu(cpu)
    ram_sort_list=map_ram(ram)
    hardisk_sort_list=map_disk(hardisk)

    #deal with the obtained information as you want