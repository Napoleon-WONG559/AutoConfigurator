import pandas as pd
import csv
import string

def remove_non_ascii(a_str):
    ascii_chars = set(string.printable)

    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )

def check_label(review_file, cpu_record, screen_record, ram_record, hardisk_record, graphic_record):
    indexes=review_file.index
    error=[]
    for ind in indexes:
        #cpu check
        if(ind!=int(cpu_record[ind][0])):
            print("cpu index error: "+ind+' record is '+cpu_record[ind][0])
            error.append(['cpu',ind])
        if(review_file.iat[ind,2] not in cpu_label[int(cpu_record[ind][2])]):
            print("cpu check error: "+ind+' is '+review_file.iat[ind,2]+' record is '+cpu_record[ind][2])
            error.append(['cpu',ind])
            
        #screen check
        if(ind!=int(screen_record[ind][0])):
            print("screen index error: "+ind+' record is '+screen_record[ind][0])
            error.append(['screen',ind])
        if(review_file.iat[ind,1] not in screen_label[int(screen_record[ind][2])]):
            print("screen check error: "+ind+' is '+review_file.iat[ind,1]+' record is '+screen_record[ind][2])
            error.append(['screen',ind])
            
        #ram check
        if(ind!=int(ram_record[ind][0])):
            print("ram index error: "+ind+' record is '+ram_record[ind][0])
            error.append(['ram',ind])
        if(review_file.iat[ind,3] not in ram_label[int(ram_record[ind][2])]):
            print("ram check error: "+ind+' is '+review_file.iat[ind,3]+' record is '+ram_record[ind][2])
            error.append(['ram',ind])
            
        #hardisk check
        if(ind!=int(hardisk_record[ind][0])):
            print("hardisk index error: "+ind+' record is '+hardisk_record[ind][0])
            error.append(['hardisk',ind])
        if(review_file.iat[ind,4] not in hardisk_label[int(hardisk_record[ind][2])]):
            print("hardisk check error: "+ind+' is '+review_file.iat[ind,4]+' record is '+hardisk_record[ind][2])
            error.append(['hardisk',ind])
            
        #graphicard check
        if(ind!=int(graphic_record[ind][0])):
            print("graphic index error: "+ind+' record is '+graphic_record[ind][0])
            error.append(['graphic',ind])
        if(review_file.iat[ind,5] not in graphic_label[int(graphic_record[ind][2])]):
            print("graphic check error: "+ind+' is '+review_file.iat[ind,5]+' record is '+graphic_record[ind][2])
            error.append(['graphic',ind])
            
    return error

if __name__ == '__main__':
    review_file=pd.read_excel('output_old_review.xlsx', index_col=0, header=0)
    
    #----------------------generate mapping file---------------------

    #graphic label mapping
    """
    graphic_label={
        0:['NVIDIA GeForce GTX 1050', 'NVIDIA GeForce GTX 1050 Ti', 'GTX 1050 Ti', 'NVIDIA GeForce GTX 1060', 'NVIDIA GeForce GTX 1070', '4GB GDDR5 NVIDIA GeForce GTX 1050', 'GTX 1050'],
        1:['AMD Radeon R4', 'radeon r5', 'AMD Radeon R5 Graphics', 'AMD Radeon R7'],
        2:['Intel UHD Graphics 620', 'Intel Iris Plus Graphics 640', 'NVIDIA GeForce 940MX'],
        3:['Intel HD Graphics 3000', 'Intel', 'Intel HD 620 graphics', 'Intel HD Graphics 500', 'Intel HD Graphics 520', 'Intel HD Graphics 620', 'Intel HD Graphics 400', 'Intel Celeron', 'Intel HD Graphics 505', 'AMD Radeon R2', 'Intel HD Graphics 5500', 'Intel HD Graphics', 'Intel?? HD Graphics 620 (up to 2.07 GB)', 'intel 620'],
        4:['Integrated', 'integrated intel hd graphics', 'integrated AMD Radeon R5 Graphics', 'Integrated Graphics', 'Integrated intel hd graphics'],
        5:[515, 4, 'FirePro W4190M', 'NONE', 'PC', 'na', 'AMD'],
    }
    """
    graphic_label={
        0:['NVIDIA GeForce GTX 1050', 'NVIDIA GeForce GTX 1050 Ti', 'GTX 1050 Ti', 'NVIDIA GeForce GTX 1060', 'NVIDIA GeForce GTX 1070', '4GB GDDR5 NVIDIA GeForce GTX 1050', 'GTX 1050', 'NVIDIA GeForce 940MX'],
        1:['AMD Radeon R2', 'AMD Radeon R4', 'radeon r5', 'AMD Radeon R5 Graphics', 'AMD Radeon R7'],
        2:['Intel UHD Graphics 620', 'Intel Iris Plus Graphics 640', 'Intel HD Graphics 3000', 'Intel', 'Intel HD 620 graphics', 'Intel HD Graphics 500', 'Intel HD Graphics 520', 'Intel HD Graphics 620', 'Intel HD Graphics 400', 'Intel Celeron', 'Intel HD Graphics 505', 'Intel HD Graphics 5500', 'Intel HD Graphics', 'Intel?? HD Graphics 620 (up to 2.07 GB)', 'intel 620'],
        3:['Integrated', 'integrated intel hd graphics', 'integrated AMD Radeon R5 Graphics', 'Integrated Graphics', 'Integrated intel hd graphics'],
        4:[515, 4, 'FirePro W4190M', 'NONE', 'PC', 'na', 'AMD'],
    }
    review_graphic_label=[]
    indexes=review_file.index
    remain=[]
    for ind in indexes:
        rev=remove_non_ascii(review_file.iat[ind,0])
        gra=review_file.iat[ind,5]
        gra_label=-1
        for key,val in graphic_label.items():
            if(gra in val):
                gra_label=key
        if(gra_label==-1):
            remain.append([ind,rev,gra])
        tmp={'index':ind,'review':rev,'graphicard_label':gra_label}
        review_graphic_label.append(tmp)
    with open('review_graphic_label_map.csv', 'w', newline='') as csvfile:
        fieldnames = ['index', 'review','graphicard_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(review_graphic_label)
    
    #screen label mapping
    screen_label={
        0:['19.5 inches'],
        1:['17.3 inches'],
        2:['15.6 inches'],
        3:['14 inches'],
        4:['13.5 inches', '13.3 inches'],
        5:['12.5 inches', '12.3 inches'],
        6:['11.6 inches'],
        7:['10.1 inches'],
    }
    review_screen_label=[]
    indexes=review_file.index
    remain=[]
    for ind in indexes:
        rev=remove_non_ascii(review_file.iat[ind,0])
        scre=review_file.iat[ind,1]
        scre_label=-1
        for key,val in screen_label.items():
            if(scre in val):
                scre_label=key
        if(scre_label==-1):
            remain.append([ind,rev,scre])
        tmp={'index':ind,'review':rev,'screen_label':scre_label}
        review_screen_label.append(tmp)
    with open('review_screen_label_map.csv', 'w', newline='') as csvfile:
        fieldnames = ['index', 'review','screen_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(review_screen_label)

    #cpu label mapping
    """
    cpu_label={
        0:['4 GHz Intel Core i7'],
        1:['3.8 GHz Intel Core i7', '3.8 GHz Core i7 Family', '3.5 GHz Intel Core i7'],
        2:['3 GHz 8032', '3.5 GHz 8032', '3 GHz AMD A Series', '3.1 GHz Intel Core i5', '3.4 GHz Intel Core i5', '3.6 GHz AMD A Series', '3.5 GHz Intel Core i5', '3 GHz'],
        3:['2.8 GHz Intel Core i7', '2.7 GHz Core i7 7500U', '2.7 GHz Core i7 2.7 GHz', '2.7 GHz Intel Core i7', '2.1 GHz Intel Core i7'],
        4:['2.2 GHz Intel Core i5', '2.3 GHz Intel Core i5', '2.6 GHz Intel Core i5', '2.5 GHz Intel Core i5', '2.5 GHz Core i5 7200U', 'Intel Core i5'],
        5:['2 GHz None', '2 GHz AMD A Series', '2.7 GHz Intel Core i3', '2.5 GHz Pentium', '2.5 GHz AMD A Series', '2.16 GHz Intel Celeron', '2.16 GHz Athlon 2650e', '2.7 GHz 8032', '2.48 GHz Intel Celeron', '2.4 GHz AMD A Series', '2 GHz Celeron D Processor 360', '2.4 GHz Intel Core i3', '2.3 GHz Intel Core i3', '2.4 GHz Core i3-540', '2.5 GHz Intel Core Duo', '2.2 GHz Intel Core i3', '2.7 GHz AMD A Series', '2.8 GHz 8032', '2.5 GHz Athlon 2650e', '2.9 GHz Intel Celeron', '2 GB', 'Celeron N3060'],
        6:['1.5 GHz', '1.8 GHz 8032', '1.8 GHz AMD E Series', '1.7 GHz', '1.1 GHz Intel Celeron', '1.6 GHz Intel Celeron', '1.6 GHz Intel Core 2 Duo', '1.7 GHz Exynos 5000 Series', '1.6 GHz Celeron N3060', '1.6 GHz AMD E Series', '1.1 GHz Pentium', '1.6 GHz', '1.6 GHz Intel Mobile CPU', '1.6 GHz Celeron N3050', '1.8 GHz Intel Core i7', '1.6 GHz Intel Core i5', 8032],
    }
    """
    cpu_label={
        0:['4 GHz Intel Core i7'],
        1:['3.8 GHz Intel Core i7', '3.8 GHz Core i7 Family', '3.5 GHz Intel Core i7', '3 GHz 8032', '3.5 GHz 8032', '3 GHz AMD A Series', '3.1 GHz Intel Core i5', '3.4 GHz Intel Core i5', '3.6 GHz AMD A Series', '3.5 GHz Intel Core i5', '3 GHz'],
        2:['2.8 GHz Intel Core i7', '2.7 GHz Core i7 7500U', '2.7 GHz Core i7 2.7 GHz', '2.7 GHz Intel Core i7', '2.1 GHz Intel Core i7', '2.2 GHz Intel Core i5', '2.3 GHz Intel Core i5', '2.6 GHz Intel Core i5', '2.5 GHz Intel Core i5', '2.5 GHz Core i5 7200U', 'Intel Core i5', '2 GHz None', '2 GHz AMD A Series', '2.7 GHz Intel Core i3', '2.5 GHz Pentium', '2.5 GHz AMD A Series', '2.16 GHz Intel Celeron', '2.16 GHz Athlon 2650e', '2.7 GHz 8032', '2.48 GHz Intel Celeron', '2.4 GHz AMD A Series', '2 GHz Celeron D Processor 360', '2.4 GHz Intel Core i3', '2.3 GHz Intel Core i3', '2.4 GHz Core i3-540', '2.5 GHz Intel Core Duo', '2.2 GHz Intel Core i3', '2.7 GHz AMD A Series', '2.8 GHz 8032', '2.5 GHz Athlon 2650e', '2.9 GHz Intel Celeron', '2 GB', 'Celeron N3060'],
        3:['1.5 GHz', '1.8 GHz 8032', '1.8 GHz AMD E Series', '1.7 GHz', '1.1 GHz Intel Celeron', '1.6 GHz Intel Celeron', '1.6 GHz Intel Core 2 Duo', '1.7 GHz Exynos 5000 Series', '1.6 GHz Celeron N3060', '1.6 GHz AMD E Series', '1.1 GHz Pentium', '1.6 GHz', '1.6 GHz Intel Mobile CPU', '1.6 GHz Celeron N3050', '1.8 GHz Intel Core i7', '1.6 GHz Intel Core i5', 8032],
    }
    review_cpu_label=[]
    indexes=review_file.index
    remain=[]
    for ind in indexes:
        rev=remove_non_ascii(review_file.iat[ind,0])
        cpu=review_file.iat[ind,2]
        c_label=-1
        for key,val in cpu_label.items():
            if(cpu in val):
                c_label=key
        if(c_label==-1):
            remain.append([ind,rev,cpu])
        tmp={'index':ind,'review':rev,'cpu_label':c_label}
        review_cpu_label.append(tmp)
    with open('review_cpu_label_map.csv', 'w', newline='') as csvfile:
        fieldnames = ['index', 'review','cpu_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(review_cpu_label)

    #ram label mapping
    ram_label={
        0:['16 GB DDR4', '16 GB LPDDR3_SDRAM', '16 GB SDRAM', '16 GB DDR SDRAM'],
        1:['12 GB', '12 GB DDR3', '12 GB DDR SDRAM'],
        2:['8 GB SDRAM DDR3', '8 GB DDR3 SDRAM', '8 GB DDR4 2666MHz', '8 GB DDR4', '8 GB LPDDR3', '8 GB DDR4 SDRAM', '8 GB DDR4_SDRAM', '8 GB 2-in1 Media Card Reader, USB 3.1, Type-C', '8 GB DDR SDRAM', '8 GB SDRAM DDR4', '8 GB ddr4', '8 GB sdram', '8 GB SDRAM', '8 GB'],
        3:['6 GB SDRAM', '6 GB', '6 GB SDRAM DDR4', '6 GB DDR SDRAM'],
        4:['4 GB LPDDR3_SDRAM', '4 GB SDRAM DDR4', '4 GB ddr3_sdram', '4 GB DDR3', '4 GB SDRAM', '4 GB', '4 GB SDRAM DDR3', '4 GB DDR4', '4 GB DDR3 SDRAM', '4 GB DDR SDRAM'],
        5:['2 GB SDRAM DDR3', '2 GB SDRAM', '2 GB DDR3L SDRAM', '2 GB DDR3 SDRAM'],
        6:['flash_memory_solid_state'],
    }
    review_ram_label=[]
    indexes=review_file.index
    remain=[]
    for ind in indexes:
        rev=remove_non_ascii(review_file.iat[ind,0])
        ram=review_file.iat[ind,3]
        r_label=-1
        for key,val in ram_label.items():
            if(ram in val):
                r_label=key
        if(r_label==-1):
            remain.append([ind,rev,ram])
        tmp={'index':ind,'review':rev,'ram_label':r_label}
        review_ram_label.append(tmp)
    with open('review_ram_label_map.csv', 'w', newline='') as csvfile:
        fieldnames = ['index', 'review','ram_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(review_ram_label)

    #hardisk label mapping
    hardisk_label={
        0:['2 TB HDD 5400 rpm'],
        1:['1 TB', '1 TB HDD 7200 rpm', '1000 GB Mechanical Hard Drive', '1000 GB Hybrid Drive', '1 TB HDD 5400 rpm', '1024 GB Mechanical Hard Drive', '1 TB serial_ata', '1 TB mechanical_hard_drive', '1128 GB Hybrid'],
        2:['500 GB HDD 5400 rpm', '500 GB mechanical_hard_drive', 'Solid State Drive, 512 GB', '512 GB SSD'],
        3:['256 GB Flash Memory Solid State', '256 GB', '256.00 SSD', '256 GB SSD', '320 GB HDD 5400 rpm'],
        4:['128 GB Flash Memory Solid State', '128 GB SSD'],
        5:['Intel', '16 GB SSD', '32 GB Flash Memory Solid State', '64 GB Flash Memory Solid State', '1 MB HDD 5400 rpm', '32 GB SSD', '64 GB SSD', '32 GB', '32 GB emmc', '16 GB flash_memory_solid_state', 'emmc', 'Flash Memory Solid State'],
    }
    review_hardisk_label=[]
    indexes=review_file.index
    remain=[]
    for ind in indexes:
        rev=remove_non_ascii(review_file.iat[ind,0])
        hard=review_file.iat[ind,4]
        hard_label=-1
        for key,val in hardisk_label.items():
            if(hard in val):
                hard_label=key
        if(hard_label==-1):
            remain.append([ind,rev,hard])
        tmp={'index':ind,'review':rev,'hardisk_label':hard_label}
        review_hardisk_label.append(tmp)
    with open('review_hardisk_label_map.csv', 'w', newline='') as csvfile:
        fieldnames = ['index', 'review','hardisk_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(review_hardisk_label)

    #----------------------check output mapping file---------------------
    cpu_record=[]
    with open("review_cpu_label_map.csv",newline='') as csvfile:
        read=csv.reader(csvfile)
        for item in read:
            cpu_record.append(item)

    screen_record=[]
    with open("review_screen_label_map.csv",newline='') as screenfile:
        read_screen=csv.reader(screenfile)
        for item in read_screen:
            screen_record.append(item)
            
    ram_record=[]
    with open("review_ram_label_map.csv",newline='') as ramfile:
        read_ram=csv.reader(ramfile)
        for item in read_ram:
            ram_record.append(item)
            
    hardisk_record=[]
    with open("review_hardisk_label_map.csv",newline='') as hardiskfile:
        read_hardisk=csv.reader(hardiskfile)
        for item in read_hardisk:
            hardisk_record.append(item)
            
    graphic_record=[]
    with open("review_graphic_label_map.csv",newline='') as graphicfile:
        read_graphic=csv.reader(graphicfile)
        for item in read_graphic:
            graphic_record.append(item)
    
    error=check_label(review_file, cpu_record[1:], screen_record[1:], ram_record[1:], hardisk_record[1:], graphic_record[1:])

    if(error==[]):
        print('no errors detected')
    else:
        print('errors detected')
        print(error)